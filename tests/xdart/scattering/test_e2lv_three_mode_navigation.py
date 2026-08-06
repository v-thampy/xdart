"""Production-shaped GI and Eiger catalog/navigation proof for E2-LV."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Event, Thread
import time

import fabio
import numpy as np
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering import test_e2lv_live_display as lv_support
from tests.xdart.scattering._e2sd_support import (
    write_motor_container,
    write_poni,
)
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import (
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
        lambda: lifecycle.phase in {RunPhase.IDLE, RunPhase.FAILED},
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
    return qapp, page, lifecycle, executor


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
        # Container members are presented one-based (source_frame_index + 1)
        # while the catalog key keeps its 0-based local_frame_label.
        assert shell.scientific.status.text() == "second.nxs · frame 1"
        assert shell.scientific.title.text() == "second.nxs · frame 1"

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
        assert shell.run_controls.readinessLabel.full_text() == (
            "Complete · 2 processed · 0 skipped · 0 pending · "
            "2 discovered"
        )
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_directory_grouped_image_footer_is_current_series_and_strip_counts_files(
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
    release_errors: list[str] = []
    original_projection = executor._frame_ready_owned

    def delayed_first_projection(run, event, image, session) -> None:
        if not projection_entered.is_set():
            projection_entered.set()
            if not release_projection.wait(20.0):
                raise AssertionError("projection release timed out")
        original_projection(run, event, image, session)

    monkeypatch.setattr(
        executor, "_frame_ready_owned", delayed_first_projection
    )

    def release_after_session_detach() -> None:
        if not projection_entered.wait(20.0):
            release_errors.append("projection did not start")
            release_projection.set()
            return
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            active = executor._active
            if active is not None and active.session is None:
                release_projection.set()
                return
            time.sleep(0.001)
        release_errors.append("session did not detach before projection")
        release_projection.set()

    releaser = Thread(target=release_after_session_detach, daemon=True)
    releaser.start()
    qapp, page, lifecycle, executor = _run_page(
        intent, executor=executor
    )
    releaser.join(timeout=1.0)
    assert not releaser.is_alive()
    assert release_errors == []
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
        assert shell.run_controls.readinessLabel.full_text() == (
            "Complete · 7 processed · 1 skipped · 0 pending · "
            "8 discovered"
        )

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
    root = Path(os.environ["XDART_TEST_DATA"]) / "eiger"
    raw = tmp_path / "raw"
    raw.mkdir()
    stems = (
        "Eiger_NbN_1_thin_test__200mdeg_scan001",
        "Eiger_NbN_2_thin_test__200mdeg_scan001",
    )
    for stem in stems:
        for suffix in ("_master.h5", "_data_000001.h5"):
            (raw / f"{stem}{suffix}").symlink_to(root / f"{stem}{suffix}")
    poni = Path(os.environ["XDART_TEST_DATA"]) / "eiger" / (
        "LaB6_detxn26_detyn6p5_eta3.poni"
    )
    intent = RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=(".h5",)),
        poni_file=str(poni),
        project_root=str(root),
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
        assert shell.run_controls.readinessLabel.full_text() == (
            "Complete · 4 processed · 0 skipped · 0 pending · "
            "4 discovered"
        )
        assert shell.scientific.raw.image.image is not None
        assert shell.scientific.cake.image.image is not None
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()
