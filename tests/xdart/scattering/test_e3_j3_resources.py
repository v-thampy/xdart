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
    live_record_store_max_items,
)
from xrd_tools.session import headless_scan as headless_scan_module
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import (
    FrozenRunConfiguration,
    RunIntent,
)
from xrd_tools.sources.selection import image_series_spec

from tests.xdart.scattering import test_e2lv_live_display as lv_support
from tests.xdart.scattering.test_e2p_rapid_navigation import (
    _DISPLAY_LIMIT,
    _HEARTBEAT_MAX_LIMIT_S,
    _HEARTBEAT_P95_LIMIT_S,
    _TinyIntegrator,
    _TrackingExecutor,
    _TrackingRecordStore,
    _TrackingScanSession,
    _accepted_admission,
    _heartbeat,
    _wait,
    _write_synthetic_series,
)


# The current-owner gate includes the exact Nexus transaction/writer graph;
# the retired 160 MiB bound covered only the mounted display with a dropping
# sink.  Retain a fixed full-route ceiling with platform headroom.
_RSS_LIMIT_BYTES = 256 * 1024**2
_SHUTDOWN_LIMIT_S = 5.0
_J3_FRAME_COUNT = 651
_SHARED_SHAPE = (384, 384)


class _SharedImageSource:
    """651 distinct frames sharing one immutable source image."""

    def __init__(
        self,
        members: tuple[Path, ...],
        lifecycle_facts: list[tuple[str, str]],
    ) -> None:
        if len(members) != _J3_FRAME_COUNT:
            raise ValueError("synthetic source requires the exact 651 members")
        self._members = members
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
                    source_path=member,
                    source_frame_index=0,
                    source_identity=str(member),
                )
                for label, member in enumerate(self._members, start=1)
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
    members = _write_synthetic_series(selected)
    poni = tmp_path / "rss.poni"
    poni.write_text("deterministic resource calibration")
    output = tmp_path / "rss.nexus"
    source_lifecycle: list[tuple[str, str]] = []
    source_references: list[
        weakref.ReferenceType[_SharedImageSource]
    ] = []
    plan_configurations: list[FrozenRunConfiguration] = []
    _TrackingRecordStore.instances.clear()
    _TrackingScanSession.references.clear()

    monkeypatch.setattr(
        executor_module, "build_admission_receipt", _accepted_admission
    )

    def open_source(_spec):
        source = _SharedImageSource(members, source_lifecycle)
        source_references.append(weakref.ref(source))
        return source

    monkeypatch.setattr(executor_module, "open_source", open_source)
    monkeypatch.setattr(
        executor_module,
        "poni_to_integrator",
        lambda _poni: _TinyIntegrator(),
    )
    native_plan = executor_module.native_int_reduction_plan

    def observed_plan(configuration):
        assert type(configuration) is FrozenRunConfiguration
        plan_configurations.append(configuration)
        plan = native_plan(configuration)
        assert plan.integration_1d is not None
        assert plan.integration_1d.npt == 2
        assert plan.integration_2d is None
        return plan

    monkeypatch.setattr(
        executor_module, "native_int_reduction_plan", observed_plan,
    )
    monkeypatch.setattr(
        executor_module, "FrameRecordStore", _TrackingRecordStore
    )
    monkeypatch.setattr(
        headless_scan_module, "ScanSession", _TrackingScanSession
    )

    intent = RunIntent(
        source_spec=image_series_spec(selected),
        poni_file=str(poni),
        project_root=str(tmp_path),
        save_path=str(output),
        processing_mode="Int 1D",
        output_mode="Overwrite",
        max_cores=1,
        bai_1d_args={"npt": 2},
        bai_2d_args={},
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
        assert len(run.resource_facts) == 1
        fact = run.resource_facts[0]
        assert residency.limits.heavy == fact.effective_display_count
        assert (
            residency.limits.live
            == live_record_store_max_items(2)
        )
        assert (
            residency.limits.browse
            == browse_publication_max_items(2)
        )

        records = _TrackingRecordStore.instances
        assert len(records) == 1
        scientific_records = records[0]
        assert len(run.display.artifacts) == 1
        artifact_owner = next(iter(run.display.artifacts.values()))
        assert artifact_owner.records is scientific_records
        assert artifact_owner.publications.labels() == tuple(
            range(1, _J3_FRAME_COUNT + 1)
        )
        allocation = artifact_owner.publications.allocation
        assert allocation is not None
        assert fact.granted_staging_count == allocation.staging_items
        assert fact.granted_record_heavy_count == allocation.record_heavy_items
        assert (
            fact.granted_publication_heavy_count
            == allocation.publication_heavy_items
        )
        assert scientific_records.item_peak <= allocation.record_items
        # J3 exercises the full large-frame/checkpoint transient route: its
        # array-bearing record population must stay within the exact staging
        # grant, while the RSS ceiling below covers every resource category.
        transient_heavy_bound = fact.granted_staging_count
        assert scientific_records.heavy_peak <= transient_heavy_bound
        assert scientific_records._heavy_labels == []
        publication_heavy = artifact_owner.publications.heavy_labels()
        assert len(publication_heavy) == min(
            _J3_FRAME_COUNT,
            fact.granted_publication_heavy_count,
        )
        assert residency.heavy == len(publication_heavy)
        record_peak = scientific_records.item_peak
        transient_record_array_peak = scientific_records.heavy_peak
        light_publication_count = len(
            artifact_owner.publications.labels()
        )
        assert len(plan_configurations) == 1
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
        assert output.is_file()
        assert output.stat().st_size > 0

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
        shutdown_deadline = shutdown_started + _SHUTDOWN_LIMIT_S
        while (
            terminal.cleanup_status is not CleanupStatus.CLEANED
            and time.perf_counter() < shutdown_deadline
        ):
            qapp.processEvents()
            time.sleep(0.005)
            terminal = page.close_workspace()
        duplicate = page.close_workspace()
        shutdown_elapsed = time.perf_counter() - shutdown_started
        assert terminal is duplicate
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert terminal.cleanup_identity is identity
        assert lifecycle.closed is True
        _TrackingRecordStore.instances.clear()
        del scientific_records
        del artifact_owner
        del publication_heavy
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
                    "transient_record_array_peak": (
                        transient_record_array_peak
                    ),
                    "transient_record_array_bound": transient_heavy_bound,
                    "light_publication_count": light_publication_count,
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
