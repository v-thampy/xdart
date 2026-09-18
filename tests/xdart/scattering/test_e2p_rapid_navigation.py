from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from threading import current_thread
from types import SimpleNamespace
import time
import weakref

import fabio
import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from tests.xdart.scattering import test_e2lv_live_display as lv_support
from tests.xdart.scattering._output_slots import written
from xrd_tools.io import read_frame_record
from xrd_tools.sources.image import TiffSeriesSource
from xrd_tools.core.staging import (
    browse_publication_max_items,
    live_record_store_max_items,
)
from xrd_tools.session import headless_scan as headless_scan_module
from xrd_tools.session.frame_record_store import FrameRecordStore
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import (
    FrozenRunConfiguration,
    RunIntent,
)
from xrd_tools.session.scan_session import ScanSession
from xrd_tools.sources.selection import image_series_spec
from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.state_machine import RunPhase


_FRAME_COUNT = 651
_DISPLAY_LIMIT = 2
# Clean runs are measured in tens of milliseconds; these subsecond ceilings
# retain ample platform headroom while rejecting a user-visible GUI stall.
_HEARTBEAT_P95_LIMIT_S = 0.25
_HEARTBEAT_MAX_LIMIT_S = 0.50


def _write_synthetic_series(selected: Path) -> tuple[Path, ...]:
    prefix = selected.stem.rsplit("_", 1)[0]
    members = tuple(
        selected.with_name(f"{prefix}_{label:04d}{selected.suffix}")
        for label in range(1, _FRAME_COUNT + 1)
    )
    for label, member in enumerate(members, start=1):
        fabio.tifimage.TifImage(
            data=np.full((2, 2), label, dtype=np.uint16)
        ).write(str(member))
    return members


def _wait(
    qapp: QtWidgets.QApplication,
    predicate,
    *,
    timeout: float = 30.0,
    diagnostic=lambda: "",
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError(
        f"production-shaped E2-P.1 run timed out: {diagnostic()}"
    )


class _TinyIntegrator:
    def __init__(self, calibrated):
        self.detector = calibrated.detector
        for name in (
            "dist", "poni1", "poni2", "rot1", "rot2", "rot3",
            "wavelength", "parallax",
        ):
            setattr(self, name, getattr(calibrated, name))

    def integrate1d(self, image, npt, *, unit, **_kwargs):
        time.sleep(0.001)
        value = float(np.asarray(image).mean())
        return SimpleNamespace(
            radial=np.linspace(0.0, 1.0, npt),
            intensity=np.full(npt, value),
            sigma=None,
            unit=unit,
        )


class _TrackingRecordStore(FrameRecordStore):
    instances: list["_TrackingRecordStore"] = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.item_peak = 0
        self.heavy_peak = 0
        type(self).instances.append(self)

    def upsert(self, record, **kwargs):
        result = super().upsert(record, **kwargs)
        self.item_peak = max(self.item_peak, len(self))
        self.heavy_peak = max(self.heavy_peak, len(self._heavy_labels))
        return result


class _TrackingScanSession(ScanSession):
    references: list[weakref.ReferenceType[ScanSession]] = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        type(self).references.append(weakref.ref(self))


class _TrackingExecutor(StandardRunExecutor):
    def __init__(self) -> None:
        super().__init__(max_display_items=_DISPLAY_LIMIT)
        self.delivered_labels: list[int] = []
        self.publication_peak = 0
        self.observed_identity = None
        self.terminal_detail = ""

    def drain_events(self):
        events = super().drain_events()
        self.delivered_labels.extend(
            event.frame_key.local_frame_label
            for event in events
            if (
                event.kind is StandardEventKind.FRAME_READY
                and event.frame_key is not None
            )
        )
        if events:
            self.observed_identity = events[-1].run_identity
        for event in events:
            if event.kind is StandardEventKind.FAILED:
                self.terminal_detail = event.detail
        run = self._exact_run(self.observed_identity)
        if run is not None:
            self.publication_peak = max(
                self.publication_peak, len(run.display.payloads)
            )
        return events

    def terminal_owners(self, run_identity) -> tuple[object, ...]:
        run = self._exact_run(run_identity)
        assert run is not None
        return (
            run.closed,
            run.cleanup_status,
            run.source,
            run.session,
            run.records,
            len(run.display.payloads),
        )


def _heartbeat(timer: QtCore.QTimer) -> tuple[list[float], object]:
    interval = 0.016
    lateness: list[float] = []
    last = time.perf_counter()

    def tick() -> None:
        nonlocal last
        current = time.perf_counter()
        lateness.append(max(0.0, current - last - interval))
        last = current

    timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
    timer.setInterval(16)
    timer.timeout.connect(tick)
    return lateness, tick


def test_public_run_delivers_651_frames_with_bounded_retention(
    monkeypatch,
    tmp_path: Path,
) -> None:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    selected = tmp_path / "tiny_0001.tif"
    members = _write_synthetic_series(selected)
    poni = tmp_path / "tiny.poni"
    poni.write_text(
        "poni_version: 2\nDetector: Detector\n"
        'Detector_config: {"pixel1": 0.0001, "pixel2": 0.0001, "max_shape": [2, 2]}\n'
        "Distance: 0.1\nPoni1: 0.01\nPoni2: 0.01\n"
        "Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1e-10\n"
    )
    output = tmp_path / "tiny.nexus"
    source_lifecycle: list[tuple[str, str]] = []
    source_references: list[weakref.ReferenceType[TiffSeriesSource]] = []
    plan_configurations: list[FrozenRunConfiguration] = []
    _TrackingRecordStore.instances.clear()
    _TrackingScanSession.references.clear()

    real_open_source = executor_module.open_source
    real_to_scan = TiffSeriesSource.to_scan

    def open_source(spec):
        source = real_open_source(spec)
        assert type(source) is TiffSeriesSource
        source_lifecycle.append(("construct", current_thread().name))
        source_references.append(weakref.ref(source))
        return source

    def to_scan(source, **kwargs):
        source_lifecycle.append(("to_scan", current_thread().name))
        return real_to_scan(source, **kwargs)

    monkeypatch.setattr(executor_module, "open_source", open_source)
    monkeypatch.setattr(TiffSeriesSource, "to_scan", to_scan)
    monkeypatch.setattr(
        TiffSeriesSource, "close",
        lambda source: source_lifecycle.append(("close", current_thread().name)),
        raising=False,
    )
    real_integrator = executor_module.poni_to_integrator
    monkeypatch.setattr(
        executor_module, "poni_to_integrator",
        lambda calibration: _TinyIntegrator(real_integrator(calibration)),
    )
    native_plan = executor_module.native_int_reduction_plan

    def observed_plan(configuration, **options):
        assert type(configuration) is FrozenRunConfiguration
        plan_configurations.append(configuration)
        # An ordinary Run asks the translation to declare its companion maps.
        assert options == {"companion_modes_2d": True}
        plan = native_plan(configuration, **options)
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
        headless_scan_module, "ScanSession", _TrackingScanSession,
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
    output = written(output, "Int 1D")
    lifecycle = ScatteringCoordinator()
    executor = _TrackingExecutor()
    page = ScatteringWorkspace(
        intents=RunIntentStore(intent),
        lifecycle=lifecycle,
        sources=FilesystemSourceAdapter(),
        executor=executor,
    )
    page_ref = weakref.ref(page)
    shell, controller = lv_support._mounted(page)
    rendered_raw: list[weakref.ReferenceType[np.ndarray]] = []
    rendered_labels: list[int] = []
    original_apply_state = shell.apply_state

    def record_apply_state(
        state, *, preserve_display: bool = False, preserve_scientific: bool = False,
    ) -> None:
        heavy = state.scientific.heavy
        if heavy is not None:
            rendered_labels.append(heavy.frame.local_frame_label)
            if heavy.raw is not None:
                rendered_raw.append(weakref.ref(heavy.raw))
        original_apply_state(
            state, preserve_display=preserve_display,
            preserve_scientific=preserve_scientific,
        )

    shell.apply_state = record_apply_state
    heartbeat_timer = QtCore.QTimer(page)
    lateness, heartbeat_slot = _heartbeat(heartbeat_timer)
    del heartbeat_slot
    page.show()
    qapp.processEvents()

    try:
        _wait(
            qapp,
            lambda: shell.run_controls.startButton.isEnabled(),
        )
        heartbeat_timer.start()
        started = time.perf_counter()
        shell.run_controls.startButton.click()
        _wait(
            qapp,
            lambda: (
                lifecycle.phase is RunPhase.IDLE
                and len(executor.delivered_labels) == _FRAME_COUNT
            ),
            diagnostic=lambda: (
                f"phase={lifecycle.phase.value}; "
                f"delivered={len(executor.delivered_labels)}; "
                f"terminal={executor.terminal_detail!r}; "
                f"{lv_support._shell_diagnostic(shell, lifecycle)}"
            ),
        )
        _wait(
            qapp,
            lambda: bool(rendered_labels)
            and rendered_labels[-1] == _FRAME_COUNT,
            diagnostic=lambda: (
                f"terminal render={rendered_labels[-1:]}; "
                f"delivered={len(executor.delivered_labels)}"
            ),
        )
        elapsed = time.perf_counter() - started
        heartbeat_timer.stop()

        assert executor.delivered_labels == list(range(1, _FRAME_COUNT + 1))
        assert rendered_labels
        assert rendered_labels[-1] == _FRAME_COUNT
        assert (
            shell.scientific.frame_selector.currentData()
            is controller.frame_keys[-1]
        )
        assert (
            controller.frame_keys[-1].local_frame_label
            == _FRAME_COUNT
        )
        assert shell.scientific.title.text() == members[-1].name
        assert executor.publication_peak <= _DISPLAY_LIMIT

        records = _TrackingRecordStore.instances
        assert len(records) == 1
        scientific_records = records[0]
        identity = controller.run_identity
        assert identity is not None
        assert identity is executor.observed_identity
        run = executor._exact_run(identity)
        assert run is not None
        assert len(run.display.artifacts) == 1
        artifact_owner = next(iter(run.display.artifacts.values()))
        assert artifact_owner.records is scientific_records
        assert artifact_owner.publications.labels() == tuple(
            range(1, _FRAME_COUNT + 1)
        )
        residency = run.display.residency_snapshot()
        assert residency.limits.live == live_record_store_max_items(2)
        assert len(run.resource_facts) == 1
        fact = run.resource_facts[0]
        assert (
            residency.limits.heavy
            == fact.effective_display_count
        )
        assert residency.limits.browse == browse_publication_max_items(2)
        assert scientific_records._max_items is None
        assert scientific_records._max_heavy_items is None
        allocation = artifact_owner.publications.allocation
        assert allocation is not None
        assert fact.granted_staging_count == allocation.staging_items
        assert fact.granted_record_heavy_count == allocation.record_heavy_items
        assert (
            fact.granted_publication_heavy_count
            == allocation.publication_heavy_items
        )
        # This route has only a 1-D result.  The allocation charges every
        # resident integrated trace to record_items; record_heavy_items is the
        # separate 2-D grant and must not be used as a proxy for these arrays.
        transient_heavy_bound = allocation.record_items
        assert allocation.record_items == residency.limits.live
        assert scientific_records.item_peak <= allocation.record_items
        assert scientific_records.heavy_peak <= transient_heavy_bound
        assert scientific_records._heavy_labels == []
        publication_heavy = artifact_owner.publications.heavy_labels()
        assert len(publication_heavy) == min(
            _FRAME_COUNT,
            fact.granted_publication_heavy_count,
        )
        assert residency.heavy == len(publication_heavy)
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
        assert len(plan_configurations) == 1
        assert len(source_references) == 1
        assert output.is_file()
        assert output.stat().st_size > 0
        for label in (1, _FRAME_COUNT):
            view = read_frame_record(output, label).active_view()
            np.testing.assert_array_equal(view.intensity_1d, np.full(2, label))
            assert Path(view.source_path) == members[label - 1]

        ordered = sorted(lateness)
        assert len(ordered) >= 8
        assert all(math.isfinite(value) and value >= 0.0 for value in ordered)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        maximum = ordered[-1]
        assert p95 < _HEARTBEAT_P95_LIMIT_S
        assert maximum < _HEARTBEAT_MAX_LIMIT_S
        (tmp_path / "e2p1-heartbeat.json").write_text(
            json.dumps(
                {
                    "frames": _FRAME_COUNT,
                    "projected_in_order": True,
                    "rendered_final": rendered_labels[-1],
                    "publication_peak": executor.publication_peak,
                    "record_peak": scientific_records.item_peak,
                    "transient_record_array_peak": (
                        scientific_records.heavy_peak
                    ),
                    "transient_record_array_bound": transient_heavy_bound,
                    "light_publication_count": len(
                        artifact_owner.publications.labels()
                    ),
                    "heartbeat_count": len(ordered),
                    "heartbeat_p95_limit_s": _HEARTBEAT_P95_LIMIT_S,
                    "heartbeat_max_limit_s": _HEARTBEAT_MAX_LIMIT_S,
                    "heartbeat_p95_lateness_s": p95,
                    "heartbeat_max_lateness_s": maximum,
                    "delivery_wall_s": elapsed,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        heartbeat_timer.stop()
        page.close_workspace()
        page.close()
        page.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(
            None, QtCore.QEvent.Type.DeferredDelete
        )
        qapp.processEvents()

    assert lifecycle.closed
    assert not any(isinstance(value, np.ndarray) for value in vars(page).values())
    session_refs = tuple(_TrackingScanSession.references)
    _TrackingRecordStore.instances.clear()
    _TrackingScanSession.references.clear()
    del heartbeat_timer
    del original_apply_state
    del record_apply_state
    del shell
    del controller
    del artifact_owner
    del publication_heavy
    del scientific_records
    del records
    del residency
    del identity
    del page
    del executor
    del run
    deadline = time.monotonic() + 2.0
    while page_ref() is not None and time.monotonic() < deadline:
        gc.collect()
        QtCore.QCoreApplication.sendPostedEvents(
            None, QtCore.QEvent.Type.DeferredDelete
        )
        qapp.processEvents()
        time.sleep(0.005)
    assert page_ref() is None
    assert all(reference() is None for reference in source_references)
    assert all(reference() is None for reference in session_refs)
    assert all(reference() is None for reference in rendered_raw)
