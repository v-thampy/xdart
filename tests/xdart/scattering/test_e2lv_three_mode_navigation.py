"""Production-shaped GI and Eiger catalog/navigation proof for E2-LV."""

from __future__ import annotations

from pathlib import Path
from threading import Event
import time

import fabio
import numpy as np
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering import test_e2lv_live_display as lv_support
from tests.xdart.scattering._e2sd_support import (
    write_motor_container,
    write_poni,
)
from tests.xdart.scattering.test_p1b_output_graph import _write_eiger
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import (
    DirectoryFileProgress,
    ShellCommand,
    ShellCommandKind,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


def _wait(
    qapp: QtWidgets.QApplication,
    predicate,
    *,
    page: ScatteringWorkspace,
    lifecycle: ScatteringCoordinator,
    timeout: float = 180.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    shell, _controller = lv_support._mounted(page)
    raise AssertionError(
        f"E2-LV mode run timed out: phase={lifecycle.phase.value}; "
        f"{lv_support._shell_diagnostic(shell, lifecycle)}"
    )


def _run_page(
    intent: RunIntent,
    *,
    executor: StandardRunExecutor | None = None,
) -> tuple[
    QtWidgets.QApplication,
    ScatteringWorkspace,
    ScatteringCoordinator,
    StandardRunExecutor,
]:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    lifecycle = ScatteringCoordinator()
    executor = executor or StandardRunExecutor(max_display_items=2)
    page = ScatteringWorkspace(
        intents=RunIntentStore(intent),
        lifecycle=lifecycle,
        sources=FilesystemSourceAdapter(),
        executor=executor,
    )
    shell, _controller = lv_support._mounted(page)
    _wait(
        qapp,
        lambda: shell.run_controls.startButton.isEnabled(),
        page=page,
        lifecycle=lifecycle,
    )
    shell.run_controls.startButton.click()
    _wait(
        qapp,
        lambda: lifecycle.phase is RunPhase.FAILED or (
            lifecycle.phase is RunPhase.IDLE
            and executor._active is not None
            and executor._active.closed
        ),
        page=page,
        lifecycle=lifecycle,
    )
    active = executor._active
    assert lifecycle.phase is RunPhase.IDLE, (
        f"status={shell.scientific.status.text()!r}; "
        f"notice={page._notice_text!r}; "
        f"progress={page._progress.detail!r}; "
        f"primary={None if active is None else active.primary!r}"
    )
    _select_terminal_acquisition(qapp, page, lifecycle, executor)
    return qapp, page, lifecycle, executor


def _select_terminal_acquisition(qapp, page, lifecycle, executor) -> None:
    """Finish the real terminal Browse handoff, then select run history."""
    shell, controller = lv_support._mounted(page)
    # A clean Run hands its final artifact to Browse asynchronously. Let that
    # real handoff settle, then explicitly inspect the retained acquisition
    # whose cross-artifact navigation these tests exercise.
    _wait(
        qapp,
        lambda: controller.browse_context is not None and (
            controller.capture_loaded_browse(
                controller.browse_context.load_request,
            ) is not None
        ),
        page=page,
        lifecycle=lifecycle,
    )
    controller.select_acquisition()
    catalog = executor.frame_catalog(controller.run_identity)
    assert catalog is not None and catalog.entries
    last = catalog.entries[-1]
    page._handle_shell_command(ShellCommand(
        ShellCommandKind.SELECT_FRAME, frame=last, frames=(last,),
    ))
    _wait(qapp, lambda: shell.scientific.frame_selector.currentData() is last,
          page=page, lifecycle=lifecycle)


def test_multi_output_gi_repeated_labels_keep_distinct_catalog_entries(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_motor_container(raw / "first.nxs")
    write_motor_container(raw / "second.nxs")
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    intent = RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=(".nxs",)),
        poni_file=str(poni),
        project_root=str(tmp_path),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
        max_cores=1,
        bai_1d_args={"npt": 8},
        bai_2d_args={"npt_rad": 8, "npt_azim": 6},
        gi=GIIntent(
            enabled=True,
            incidence_motor="Manual",
            th_val=0.2,
            mode_1d="q_total",
            mode_2d="qip_qoop",
        ),
    )
    qapp, page, lifecycle, executor = _run_page(intent)
    shell, controller = lv_support._mounted(page)
    try:
        identity = controller.run_identity
        assert identity is not None
        catalog = executor.frame_catalog(identity)
        assert catalog is not None
        assert len(controller.frame_keys) == len(catalog.entries)
        assert all(
            actual is expected
            for actual, expected in zip(
                controller.frame_keys, catalog.entries, strict=True
            )
        )
        assert [key.local_frame_label for key in catalog.entries] == [0, 0]
        assert [key.work_ordinal for key in catalog.entries] == [1, 2]
        assert len({key.source_scan for key in catalog.entries}) == 2
        assert len({key.artifact for key in catalog.entries}) == 2
        selector = shell.scientific.frame_selector
        assert selector.count() == 1
        assert tuple(
            selector.itemData(index)
            for index in range(selector.count())
        ) == catalog.entries[1:]
        assert shell.scientific.progress.text() == "1/1"
        # This raw .nxs member keeps its filename in the footer; its exact
        # zero-based container label remains in the catalog key.
        assert shell.scientific.status.text() == "second.nxs"
        assert shell.scientific.title.text() == "second.nxs"

        first = catalog.entries[0]
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SELECT_FRAME,
            frame=first,
            frames=(first,),
        ))
        _wait(
            qapp,
            lambda: selector.currentData() is first,
            page=page,
            lifecycle=lifecycle,
        )
        assert selector.count() == 1
        assert selector.itemData(0) is first
        assert shell.scientific.progress.text() == "1/1"
        assert page._progress.directory_files == DirectoryFileProgress(2, 0, 0, 2)
        assert shell.run_controls.readinessLabel.full_text().startswith("Complete · 2 Frames · ")
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_directory_grouped_image_footer_is_current_series_and_progress_counts_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for prefix, count, image_type in (
        ("alpha", 4, "tif"),
        ("beta", 3, "edf"),
    ):
        for index in range(1, count + 1):
            data = np.full((195, 487), index, dtype=np.uint16)
            image = (
                fabio.tifimage.TifImage(data=data)
                if image_type == "tif"
                else fabio.edfimage.EdfImage(data=data)
            )
            image.write(str(raw / f"{prefix}_{index:04d}.{image_type}"))
    (raw / "corrupt_0001.tif").write_bytes(b"not a TIFF")
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    intent = RunIntent(
        source_spec=DirectorySourceSpec(
            raw, suffixes=(".tif", ".edf")
        ),
        poni_file=str(poni),
        project_root=str(tmp_path),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
        max_cores=1,
        bai_1d_args={"npt": 8},
        bai_2d_args={"npt_rad": 8, "npt_azim": 6},
    )
    executor = StandardRunExecutor(max_display_items=2)
    projection_entered = Event()
    release_projection = Event()
    terminal_before_projection = []
    original_projection = executor._frame_ready_owned
    original_finish_projection = executor._finish_display_projection

    def delayed_first_projection(run, event, image, session) -> None:
        if not projection_entered.is_set():
            projection_entered.set()
            if not release_projection.wait(20.0):
                raise AssertionError("projection release timed out")
        original_projection(run, event, image, session)

    monkeypatch.setattr(
        executor, "_frame_ready_owned", delayed_first_projection
    )

    def drain_after_writer_finish(run) -> None:
        if not release_projection.is_set():
            try:
                assert projection_entered.wait(20.0)
                # Writer settlement precedes display drain. The session is
                # retained until these queued publications have been drained.
                assert run.session is not None
                terminal_before_projection.append(run.session.terminal_result)
            finally:
                release_projection.set()
        original_finish_projection(run)

    monkeypatch.setattr(executor, "_finish_display_projection", drain_after_writer_finish)
    qapp, page, lifecycle, executor = _run_page(
        intent, executor=executor
    )
    assert len(terminal_before_projection) == 1
    assert terminal_before_projection[0].commit_identity is not None
    shell, controller = lv_support._mounted(page)
    try:
        identity = controller.run_identity
        assert identity is not None
        catalog = executor.frame_catalog(identity)
        assert catalog is not None
        assert len(catalog.entries) == 7
        assert len(controller.frame_keys) == 7
        assert len({key.artifact for key in catalog.entries}) == 2

        selector = shell.scientific.frame_selector
        assert selector.count() == 3
        assert shell.scientific.progress.text() == "3/3"
        assert shell.scientific.status.text() == "beta_0003.edf"
        assert page._progress.directory_files == DirectoryFileProgress(7, 1, 0, 8)
        assert shell.run_controls.readinessLabel.full_text().startswith("Complete · 7 Frames · ")

        first = catalog.entries[0]
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SELECT_FRAME,
            frame=first,
            frames=(first,),
        ))
        _wait(
            qapp,
            lambda: selector.currentData() is first,
            page=page,
            lifecycle=lifecycle,
        )
        assert selector.count() == 4
        assert shell.scientific.progress.text() == "1/4"
        # The complete run history remains available behind the scan-local
        # footer for browser selection, hydration, and log accounting.
        assert len(executor.frame_catalog(identity).entries) == 7
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_eiger_outputs_with_repeated_local_labels_remain_navigable(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    stems = (
        "Eiger_NbN_1_thin_test__200mdeg_scan001",
        "Eiger_NbN_2_thin_test__200mdeg_scan001",
    )
    for stem in stems:
        _write_eiger(raw / f"{stem}_master.h5", raw / f"{stem}_data_000001.h5", 5)
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    intent = RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=(".h5",)),
        poni_file=str(poni),
        project_root=str(tmp_path),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
        max_cores=1,
        bai_1d_args={"npt": 8, "method": "csr"},
        bai_2d_args={
            "npt_rad": 8,
            "npt_azim": 6,
            "method": "csr",
        },
    )
    qapp, page, lifecycle, executor = _run_page(intent)
    shell, controller = lv_support._mounted(page)
    try:
        identity = controller.run_identity
        assert identity is not None
        catalog = executor.frame_catalog(identity)
        assert catalog is not None
        assert len(controller.frame_keys) == len(catalog.entries)
        assert all(
            actual is expected
            for actual, expected in zip(
                controller.frame_keys, catalog.entries, strict=True
            )
        )
        assert len(catalog.entries) == 10
        assert [key.local_frame_label for key in catalog.entries] == (
            list(range(5)) + list(range(5))
        )
        assert len({key.source_scan for key in catalog.entries}) == 2
        assert len({key.artifact for key in catalog.entries}) == 2
        selector = shell.scientific.frame_selector
        assert selector.count() == 5
        assert tuple(
            selector.itemData(index)
            for index in range(selector.count())
        ) == catalog.entries[5:]
        assert shell.scientific.progress.text() == "5/5"

        first = catalog.entries[0]
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SELECT_FRAME,
            frame=first,
            frames=(first,),
        ))
        _wait(
            qapp,
            lambda: (
                selector.currentData() is first
                and shell.scientific.title.text().endswith(
                    "· frame 1"
                )
            ),
            page=page,
            lifecycle=lifecycle,
        )
        assert selector.count() == 5
        assert tuple(
            selector.itemData(index)
            for index in range(selector.count())
        ) == catalog.entries[:5]
        assert shell.scientific.progress.text() == "1/5"
        assert page._progress.directory_files == DirectoryFileProgress(4, 0, 0, 4)
        assert shell.run_controls.readinessLabel.full_text().startswith("Complete · 10 Frames · ")
        assert shell.scientific.raw.image.image is not None
        assert shell.scientific.cake.image.image is not None
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()
