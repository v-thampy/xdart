"""OS-level resource evidence for the mounted E3 public route."""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
from threading import current_thread
import time
import weakref

import fabio.tifimage
import numpy as np
import psutil
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.adapters import (
    run_executor as executor_module,
)
from xdart.gui.tabs.scattering.adapters.source import (
    FilesystemSourceAdapter,
)
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xrd_tools.core.scan import Scan, ScanFrame
from xrd_tools.core.staging import (
    browse_publication_max_items,
    heavy_window,
    live_record_store_max_items,
)
from xrd_tools.reduction import Integration1DPlan, ReductionPlan
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec

from tests.xdart.scattering import test_e2lv_live_display as lv_support
from tests.xdart.scattering.test_e2p_rapid_navigation import (
    _DISPLAY_LIMIT,
    _DroppingSink,
    _HEARTBEAT_MAX_LIMIT_S,
    _HEARTBEAT_P95_LIMIT_S,
    _TinyIntegrator,
    _TrackingExecutor,
    _TrackingRecordStore,
    _TrackingScanSession,
    _accepted_admission,
    _heartbeat,
    _wait,
)


_RSS_LIMIT_BYTES = 160 * 1024**2
_SHUTDOWN_LIMIT_S = 5.0
_J3_FRAME_COUNT = 651
_SHARED_SHAPE = (384, 384)


class _SharedImageSource:
    """651 distinct frames sharing one immutable source image."""

    def __init__(
        self,
        selected: Path,
        lifecycle_facts: list[tuple[str, str]],
    ) -> None:
        self._selected = selected
        self._lifecycle_facts = lifecycle_facts
        image = np.arange(
            int(np.prod(_SHARED_SHAPE)), dtype=np.float32
        ).reshape(_SHARED_SHAPE)
        image.setflags(write=False)
        self._image = image
        lifecycle_facts.append(("construct", current_thread().name))

    def to_scan(self, *, poni, integrator, output_path):
        self._lifecycle_facts.append(
            ("to_scan", current_thread().name)
        )
        return Scan(
            "rss-651",
            [
                ScanFrame(
                    label,
                    image=self._image,
                    source_path=self._selected,
                    source_frame_index=label,
                )
                for label in range(1, _J3_FRAME_COUNT + 1)
            ],
            poni=poni,
            integrator=integrator,
            output_path=output_path,
        )

    def close(self) -> None:
        self._lifecycle_facts.append(
            ("close", current_thread().name)
        )


def test_j3_651_frame_os_rss_remains_bounded_at_public_mount(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    selected = tmp_path / "rss_0001.tif"
    fabio.tifimage.TifImage(
        data=np.ones((2, 2), dtype=np.uint16)
    ).write(str(selected))
    poni = tmp_path / "rss.poni"
    poni.write_text("deterministic resource calibration")
    output = tmp_path / "rss.nxs"
    source_lifecycle: list[tuple[str, str]] = []
    source_references: list[
        weakref.ReferenceType[_SharedImageSource]
    ] = []
    sinks: list[_DroppingSink] = []
    _TrackingRecordStore.instances.clear()
    _TrackingScanSession.references.clear()

    monkeypatch.setattr(
        executor_module, "build_admission_receipt", _accepted_admission
    )

    def open_source(_spec):
        source = _SharedImageSource(selected, source_lifecycle)
        source_references.append(weakref.ref(source))
        return source

    monkeypatch.setattr(executor_module, "open_source", open_source)
    monkeypatch.setattr(
        executor_module,
        "poni_to_integrator",
        lambda _poni: _TinyIntegrator(),
    )
    monkeypatch.setattr(
        executor_module,
        "build_native_int_reduction_plan_from_args",
        lambda *_args, **_kwargs: ReductionPlan(
            integration_1d=Integration1DPlan(npt=2),
            integration_2d=None,
        ),
    )

    def sink_factory(*args, **kwargs):
        sink = _DroppingSink(*args, **kwargs)
        sinks.append(sink)
        return sink

    monkeypatch.setattr(executor_module, "NexusSink", sink_factory)
    monkeypatch.setattr(
        executor_module, "FrameRecordStore", _TrackingRecordStore
    )
    monkeypatch.setattr(
        executor_module, "ScanSession", _TrackingScanSession
    )

    intent = RunIntent(
        source_spec=image_series_spec(selected),
        poni_file=str(poni),
        project_root=str(tmp_path),
        save_path=str(output),
        output_mode="Overwrite",
        max_cores=1,
    )
    lifecycle = ScatteringCoordinator()
    executor = _TrackingExecutor()
    page = ScatteringWorkspace(
        intents=RunIntentStore(intent),
        lifecycle=lifecycle,
        sources=FilesystemSourceAdapter(),
        executor=executor,
    )
    shell, controller = lv_support._mounted(page)
    process = psutil.Process(os.getpid())
    heartbeat_timer = QtCore.QTimer(page)
    rss_timer = QtCore.QTimer(page)
    rss_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
    rss_timer.setInterval(5)
    rss_samples: list[int] = []

    def sample_rss() -> None:
        rss_samples.append(process.memory_info().rss)

    rss_timer.timeout.connect(sample_rss)
    page.show()
    qapp.processEvents()
    _wait(qapp, lambda: shell.run_controls.startButton.isEnabled())
    gc.collect()
    qapp.processEvents()
    baseline_rss = process.memory_info().rss
    rss_samples.append(baseline_rss)
    lateness, heartbeat_slot = _heartbeat(heartbeat_timer)
    del heartbeat_slot
    run = None
    terminal = None
    try:
        heartbeat_timer.start()
        rss_timer.start()
        started = time.perf_counter()
        shell.run_controls.startButton.click()
        _wait(
            qapp,
            lambda: (
                lifecycle.phase is RunPhase.IDLE
                and len(executor.delivered_labels) == _J3_FRAME_COUNT
            ),
            timeout=60.0,
            diagnostic=lambda: (
                f"phase={lifecycle.phase.value}; "
                f"delivered={len(executor.delivered_labels)}; "
                f"{lv_support._shell_diagnostic(shell, lifecycle)}"
            ),
        )
        qapp.processEvents()
        elapsed = time.perf_counter() - started
        heartbeat_timer.stop()
        rss_timer.stop()
        gc.collect()
        qapp.processEvents()
        run_terminal_rss = process.memory_info().rss
        rss_samples.append(run_terminal_rss)
        peak_rss = max(rss_samples)
        peak_delta = peak_rss - baseline_rss
        run_terminal_delta = run_terminal_rss - baseline_rss

        identity = controller.run_identity
        assert identity is not None
        run = executor._exact_run(identity)
        assert run is not None
        assert run.worker is not None and not run.worker.is_alive()
        assert run.display.hydration_thread is None
        assert executor.delivered_labels == list(
            range(1, _J3_FRAME_COUNT + 1)
        )
        assert executor.publication_peak <= _DISPLAY_LIMIT
        residency = run.display.residency_snapshot()
        assert residency.limits.heavy == 8 == heavy_window()
        assert (
            residency.limits.live
            == live_record_store_max_items(0)
        )
        assert (
            residency.limits.browse
            == browse_publication_max_items(0)
        )

        records = _TrackingRecordStore.instances
        assert len(records) == 2
        scientific_records, light_records = records
        assert scientific_records.item_peak <= residency.limits.live + 1
        assert scientific_records.heavy_peak <= 9
        assert light_records.item_peak == _J3_FRAME_COUNT
        record_peak = scientific_records.item_peak
        heavy_record_peak = scientific_records.heavy_peak
        light_record_peak = light_records.item_peak
        assert len(source_references) == 1
        assert len(_TrackingScanSession.references) == 1
        session_references = tuple(_TrackingScanSession.references)
        assert executor.terminal_owners(identity) == (
            True,
            CleanupStatus.CLEANED,
            None,
            None,
            None,
            _DISPLAY_LIMIT,
        )
        assert source_lifecycle == [
            ("construct", "scattering-standard"),
            ("to_scan", "scattering-standard"),
            ("close", "scattering-standard"),
        ]
        assert len(sinks) == 1
        assert (
            sinks[0].begin_calls,
            sinks[0].write_calls,
            sinks[0].finish_calls,
            sinks[0].abort_calls,
        ) == (1, _J3_FRAME_COUNT, 1, 0)

        ordered = sorted(lateness)
        assert len(ordered) >= 8
        assert all(
            math.isfinite(value) and value >= 0.0
            for value in ordered
        )
        p95 = ordered[
            min(len(ordered) - 1, int(0.95 * len(ordered)))
        ]
        maximum = ordered[-1]
        assert p95 < _HEARTBEAT_P95_LIMIT_S
        assert maximum < _HEARTBEAT_MAX_LIMIT_S
        assert peak_delta < _RSS_LIMIT_BYTES
        assert run_terminal_delta < _RSS_LIMIT_BYTES

        shutdown_started = time.perf_counter()
        terminal = page.close_workspace()
        duplicate = page.close_workspace()
        shutdown_elapsed = time.perf_counter() - shutdown_started
        assert terminal is duplicate
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert terminal.cleanup_identity is identity
        assert lifecycle.closed is True
        _TrackingRecordStore.instances.clear()
        del scientific_records
        del light_records
        del records
        gc.collect()
        qapp.processEvents()
        post_close_rss = process.memory_info().rss
        post_close_delta = post_close_rss - baseline_rss
        assert shutdown_elapsed < _SHUTDOWN_LIMIT_S
        assert post_close_delta < _RSS_LIMIT_BYTES
        assert all(
            reference() is None for reference in source_references
        )
        assert all(
            reference() is None
            for reference in session_references
        )

        (tmp_path / "j3-resource-evidence.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "session_file": os.environ["XDART_SESSION_FILE"],
                    "frames": _J3_FRAME_COUNT,
                    "shared_source_bytes": int(
                        np.prod(_SHARED_SHAPE)
                        * np.dtype(np.float32).itemsize
                    ),
                    "rss_baseline_bytes": baseline_rss,
                    "rss_peak_bytes": peak_rss,
                    "rss_run_terminal_bytes": run_terminal_rss,
                    "rss_post_close_bytes": post_close_rss,
                    "rss_peak_delta_bytes": peak_delta,
                    "rss_run_terminal_delta_bytes": run_terminal_delta,
                    "rss_post_close_delta_bytes": post_close_delta,
                    "rss_limit_bytes": _RSS_LIMIT_BYTES,
                    "rss_samples": len(rss_samples),
                    "heartbeat_count": len(ordered),
                    "heartbeat_p95_lateness_s": p95,
                    "heartbeat_max_lateness_s": maximum,
                    "publication_peak": executor.publication_peak,
                    "record_peak": record_peak,
                    "heavy_record_peak": heavy_record_peak,
                    "light_record_peak": light_record_peak,
                    "delivery_wall_s": elapsed,
                    "shutdown_wall_s": shutdown_elapsed,
                    "shutdown_limit_s": _SHUTDOWN_LIMIT_S,
                    "cleanup_status": terminal.cleanup_status.value,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        heartbeat_timer.stop()
        rss_timer.stop()
        page.close_workspace()
        page.close()
        page.deleteLater()
        qapp.processEvents()
        _TrackingRecordStore.instances.clear()
        _TrackingScanSession.references.clear()
