"""Frozen production-shaped oracle for the E2-LV responsibility split."""

from __future__ import annotations

import gc
from pathlib import Path
from threading import Event
import time

import fabio.tifimage
import numpy as np
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering import test_e2lv_live_display as lv_support
from tests.xdart.scattering._e2sd_support import (
    write_motor_container,
    write_poni,
)
from tests.xdart.scattering.test_e2lv_three_mode_navigation import _run_page, _wait
from xdart.gui.tabs.scattering.controls_inventory import THRESHOLD_MIN
from xdart.gui.tabs.scattering.display_values import (
    DisplayFrameKey,
    DisplayNavigationDelta,
    StandardEventKind,
    StandardRunEvent,
)
from xdart.gui.tabs.scattering.events import CleanupStatus, RunIdentity
from xdart.gui.tabs.scattering.shell_widgets import CompactFrameSelector
from xdart.gui.widgets.controls_panel import RangeRow
from xrd_tools.core.staging import (
    browse_publication_max_items,
    heavy_window,
    live_record_store_max_items,
)
from xrd_tools.io import read_frame_record
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


def test_multi_output_run_has_one_aggregate_residency_budget(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # The real executor must fund its sixteen-frame semantic checkpoint.
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "16")
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
    # Ensure Qt wrappers retired by earlier GUI cases are collected on the
    # GUI thread, before this production-shaped run starts admission I/O.
    gc.collect()
    qapp, page, _lifecycle, executor = _run_page(intent)
    try:
        run = executor._active
        assert run is not None
        owners = tuple(run.display.artifacts.values())
        snapshot = run.display.residency_snapshot()
        assert len(owners) == 2
        assert snapshot.limits.heavy == heavy_window()
        assert snapshot.limits.live == live_record_store_max_items(8)
        assert snapshot.limits.browse == browse_publication_max_items(8)
        assert snapshot.limits.thumbnails == 512
        assert all(owner.records._max_heavy_items is None for owner in owners)
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_catalog_append_and_selector_updates_are_linear(
    monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.display_runtime import RunDisplayState

    original_hash = DisplayFrameKey.__hash__
    calls = 0

    def counted_hash(self):
        nonlocal calls
        calls += 1
        return original_hash(self)

    monkeypatch.setattr(DisplayFrameKey, "__hash__", counted_hash)
    identity = RunIdentity(1, "catalog-cost")
    state = RunDisplayState(identity, max_payload_items=2)
    keys = [
        state.append_navigation(
            "scan", "artifact", ordinal
        ).appended
        for ordinal in range(1, 10_001)
    ]
    assert calls <= len(keys) * 10
    assert len(state.catalog_snapshot().entries) == 10_000

    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    selector = CompactFrameSelector()
    clears = 0
    original_clear = selector.clear

    def counted_clear() -> None:
        nonlocal clears
        clears += 1
        original_clear()

    monkeypatch.setattr(selector, "clear", counted_clear)
    try:
        for key in keys[:500]:
            selector.add_frame(
                str(key.local_frame_label),
                key,
                f"{key.source_scan}:{key.local_frame_label}",
            )
        selector.setCurrentIndex(499)
        assert selector.count() == 500
        assert selector.currentData() is keys[499]
        assert clears <= 1
    finally:
        selector.close()
        selector.deleteLater()
        qapp.processEvents()


def test_live_and_durable_value_mask_honor_operator_toggle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def write_saturated_tiff(path: Path, *, offset: int = 0) -> None:
        raw = np.arange(8, dtype=np.uint32).reshape(2, 4)
        raw[0, 1] = np.iinfo(np.uint32).max
        fabio.tifimage.TifImage(data=raw).write(str(path))

    monkeypatch.setattr(lv_support, "_write_tiff", write_saturated_tiff)
    for enabled in (True, False):
        case = tmp_path / ("enabled" if enabled else "disabled")
        case.mkdir()

        with monkeypatch.context() as scoped:
            qapp, page, lifecycle, executor, output = lv_support._standard_page(
                scoped, case, labels=(1,), mask=None
            )
            shell, _controller = lv_support._mounted(page)
            try:
                lv_support._wait(
                    qapp,
                    lambda: shell.run_controls.startButton.isEnabled(),
                )
                threshold = next(
                    row for row in shell.controls.findChildren(RangeRow)
                    if tuple(row._low_path) == THRESHOLD_MIN
                )
                assert threshold._toggle[1].isChecked()
                if not enabled:
                    threshold._toggle[1].click()
                    lv_support._wait(
                        qapp,
                        lambda: shell.run_controls.startButton.isEnabled(),
                    )
                shell.run_controls.startButton.click()
                lv_support._completed_acquisition(qapp, page, lifecycle, executor)
                rendered = shell.scientific.raw.image.image
                assert rendered is not None
                assert bool(np.isnan(rendered[1, 1])) is enabled

                view = read_frame_record(output, 1).active_view()
                assert view.thumbnail is not None
                assert bool(np.isnan(view.thumbnail[0, 1])) is enabled
                assert view.mask_baked is enabled
            finally:
                page.close_workspace()
                page.deleteLater()
                qapp.processEvents()


def test_thumbnail_fallback_renders_once_without_rehydration_loop(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "16")
    qapp, page, lifecycle, executor, output = lv_support._standard_page(
        monkeypatch, tmp_path, labels=tuple(range(1, 19)), mask=None
    )
    from xdart.gui.tabs.scattering import hydration_transport

    calls: list[int] = []
    real_reader = hydration_transport.read_frame_preview

    def traced_reader(read_key, **kwargs):
        calls.append(int(read_key.frame_identity))
        return real_reader(read_key, **kwargs)

    shell, controller = lv_support._mounted(page)
    try:
        lv_support._wait(
            qapp,
            lambda: shell.run_controls.startButton.isEnabled(),
        )
        shell.run_controls.startButton.click()
        lv_support._completed_acquisition(qapp, page, lifecycle, executor)
        assert output.is_file()
        monkeypatch.setattr(
            hydration_transport, "read_frame_preview", traced_reader
        )
        (tmp_path / "tiny_0001.tif").unlink()
        first = next(
            frame
            for frame in controller.frame_keys
            if frame.local_frame_label == 1
        )
        lv_support._select_exact_frame(shell, first)
        lv_support._wait(
            qapp,
            lambda: (
                shell.scientific.frame_selector.currentData() is first
                and shell.scientific.title.text() == "tiny_0001.tif"
            ),
            timeout=2.0,
        )
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.002)
        assert shell.scientific.raw.image.image is not None
        assert calls == [1]
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_page_close_invalidates_latched_hydration_and_exact_close_retries(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "16")
    qapp, page, lifecycle, executor, _output = lv_support._standard_page(
        monkeypatch, tmp_path, labels=tuple(range(1, 19)), mask=None
    )
    from xdart.gui.tabs.scattering import hydration_transport

    entered = Event()
    release = Event()
    real_reader = hydration_transport.read_frame_preview

    def latched_reader(read_key, **kwargs):
        entered.set()
        assert release.wait(5.0)
        return real_reader(read_key, **kwargs)

    worker = None
    shell, controller = lv_support._mounted(page)
    try:
        lv_support._wait(
            qapp,
            lambda: shell.run_controls.startButton.isEnabled(),
        )
        shell.run_controls.startButton.click()
        lv_support._completed_acquisition(qapp, page, lifecycle, executor)
        identity = controller.run_identity
        assert identity is not None
        run = executor._exact_run(identity)
        assert run is not None
        # Deliver the completed terminal Browse work before latching the
        # distinct acquisition request whose post-close events are forbidden.
        lv_support._wait(qapp, lambda: run.display.hydration_thread is None)
        page._drain_executor()
        monkeypatch.setattr(
            hydration_transport, "read_frame_preview", latched_reader
        )
        executor._join_timeout = 0.01

        first = next(
            frame
            for frame in controller.frame_keys
            if frame.local_frame_label == 1
        )
        lv_support._select_exact_frame(shell, first)
        assert entered.wait(2.0)
        page.close_workspace()
        pending = run.cleanup_status
        worker = run.display.hydration_thread
        release.set()
        if worker is not None:
            worker.join(2.0)
        events = executor.drain_events()
        retried = executor.close(identity)

        assert pending is CleanupStatus.CLEANUP_PENDING
        assert worker is not None and not worker.is_alive()
        assert not any(
            event.kind is StandardEventKind.DISPLAY_READY for event in events
        ), [(event.kind, event.frame_key) for event in events]
        assert retried.cleanup_status is CleanupStatus.CLEANED
    finally:
        release.set()
        if worker is not None:
            worker.join(2.0)
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_delayed_frame_event_routes_its_exact_artifact_key(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_motor_container(raw / "first.nxs")
    write_motor_container(raw / "second.nxs")
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    qapp, page, _lifecycle, executor = _run_page(
        RunIntent(
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
    )
    shell, controller = lv_support._mounted(page)
    try:
        identity = controller.run_identity
        assert identity is not None
        catalog = executor.frame_catalog(identity)
        assert catalog is not None and len(catalog.entries) == 2
        delayed = catalog.entries[0]
        newest = catalog.entries[1]
        assert delayed.artifact != newest.artifact
        assert delayed.local_frame_label == newest.local_frame_label

        executor._events.put(
            StandardRunEvent(
                identity,
                StandardEventKind.FRAME_READY,
                completed=2,
                total=2,
                artifact=delayed.artifact,
                frame_key=delayed,
                navigation_delta=DisplayNavigationDelta(delayed),
            )
        )
        page._drain_executor()

        # Cold cross-artifact hydration keeps the outgoing plot until the exact
        # replacement is ready; event acceptance alone is not a paint receipt,
        # and neither is selector acceptance: the status follows the selection
        # on every apply, the title only once the replacement presentation
        # lands, so a slow host can show the new status under the old title.
        _wait(
            qapp,
            lambda: (
                shell.scientific.frame_selector.currentData() is delayed
                and shell.scientific.title.text()
                == shell.scientific.status.text()
            ),
            page=page,
            lifecycle=_lifecycle,
        )
        assert shell.scientific.frame_selector.currentData() is delayed
        status = shell.scientific.status.text()
        assert status == shell.scientific.title.text()
        assert "first.nxs" in status
        assert "second.nxs" not in status
        assert delayed.artifact not in status
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()
