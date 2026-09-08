from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from threading import Event
import time
from types import SimpleNamespace

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from xrd_tools.core import FrameRecord, FrameView
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xrd_tools.sources.selection import image_series_spec
from xdart.gui.tabs.scattering.contracts import (
    SourceCapture, SourceObservation, SourceObservationRequest, SourceObservationStatus,
)
from xdart.gui.tabs.scattering.controls_inventory import INT_1D_AXIS
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import (
    DisplayFrameKey,
    DisplayNavigationDelta,
    StandardDisplayPayload,
    StandardEventKind,
    StandardRunEvent,
    StandardTerminalTiming,
)
import xdart.gui.tabs.scattering.display_values as display_values_module
from xdart.gui.tabs.scattering.events import (
    CleanupStatus, ExecutorAccepted, ExecutorClosed, ExecutorStartFailed,
    PreflightAccepted, RunIdentity,
)
from xdart.gui.tabs.scattering.display_retirement import (
    DisplayRetirementReceipt,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.processed_browser import (
    ProcessedBrowserOwner,
    TerminalBrowseHandoff,
)
import xdart.gui.tabs.scattering.page as page_module
from xdart.gui.tabs.scattering.performance_diagnostics import (
    PerformanceDiagnosticsDialog,
    PerformanceDiagnosticsValues,
)
from xdart.gui.tabs.scattering.shell_values import (
    FrameNavigationProjection,
    ShellCommand,
    ShellCommandKind,
)
from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xdart.modules.display_context import ContextKind
from xdart.modules.frame_publication import FramePublication
from tests.xdart.scattering._admission import (
    ImmediateAdmission,
    admission_for,
)


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Sources:
    def __init__(self) -> None:
        self.epoch = 0

    def capture(self, source, request_id):
        self.epoch += 1
        return SourceCapture(request_id, self.epoch, source)

    def cancel(self, _request_id) -> None:
        return None

    def observe(self, request: SourceObservationRequest) -> SourceObservation:
        return SourceObservation(request.observation_id, request.intent_revision, request.source,
                                 SourceObservationStatus.AVAILABLE, "frame.tif", True, False)

    def cancel_observation(self, _observation_id: int) -> None:
        return None

    def publish_motor_knowledge(self, _observation) -> None:
        return None

    def project_motor_knowledge(self, _source, _fingerprint=None):
        return None


class _Executor(ImmediateAdmission):
    def __init__(self) -> None:
        self.events: list[object] = []
        self.stop_error: Exception | None = None
        self.start_calls = 0
        self.last_identity: RunIdentity | None = None
        self.last_configuration = None

    def begin_admission(self, capture):
        token = super().begin_admission(capture)
        identity = self.last_identity
        if identity is not None:
            self._test_admission = (
                token,
                replace(
                    admission_for(capture),
                    display_retirement=DisplayRetirementReceipt(
                        identity, CleanupStatus.CLEANED
                    ),
                ),
            )
        return token

    def start(self, _configuration, _source, run_identity, _admission):
        self.start_calls += 1
        self.last_identity = run_identity
        self.last_configuration = _configuration
        return ExecutorAccepted(run_identity)

    def stop(self, _run_identity) -> None:
        if self.stop_error is not None:
            raise self.stop_error

    def close(self, run_identity):
        return ExecutorClosed(run_identity, CleanupStatus.CLEANED)

    def pause(self, _run_identity) -> None:
        return None

    def resume(self, _run_identity) -> None:
        return None

    def drain_events(self):
        events, self.events = tuple(self.events), []
        return events

def _active_page(
    executor: _Executor,
    *,
    output_mode: str = "Overwrite",
    processing_mode: str = "Int 2D",
    project_root: str = "",
) -> tuple[ScatteringWorkspace, ScatteringCoordinator, RunIdentity]:
    lifecycle = ScatteringCoordinator()
    request = lifecycle.begin_start().request_id
    assert request is not None
    configuration = RunIntent(
        output_mode=output_mode,
        processing_mode=processing_mode,
        project_root=project_root,
    ).freeze()
    identity = lifecycle.preflight_accepted(PreflightAccepted(request, configuration)).run_identity
    assert identity is not None
    assert lifecycle.executor_accepted(ExecutorAccepted(identity)).phase is RunPhase.RUNNING
    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                source_spec=image_series_spec(Path("frame_0001.tif")),
                poni_file="calibration.poni",
                save_path="output.nxs",
                output_mode=output_mode,
                processing_mode=processing_mode,
                project_root=project_root,
            )
        ),
        lifecycle=lifecycle,
        sources=_Sources(),
        executor=executor,
    )
    return page, lifecycle, identity


def _dispose(page: ScatteringWorkspace, qapp: QtWidgets.QApplication) -> None:
    page.close_workspace()
    page.deleteLater()
    qapp.processEvents()


def _shell(page: ScatteringWorkspace) -> ScatteringWorkspaceShell:
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    return shell


def _wait_until(qapp, predicate, *, timeout=5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("asynchronous test owner did not settle")


def _paced_frame_events(
    page,
    executor,
    identity,
    count,
    *,
    processing_mode="Int 2D",
    project_root="",
):
    from tests.xdart.scattering.test_e3_context_contract import _acquisition

    _, acquisition = _acquisition(
        configuration=RunIntent(
            output_mode="Overwrite",
            processing_mode=processing_mode,
            project_root=project_root,
        ).freeze(),
        identity=identity,
    )
    executor.acquisition_context = lambda candidate: acquisition if candidate is identity else None
    acquisition.publication_store.catalog.resize(max(16, count))
    page._context_controller.adopt_acquisition(identity)
    display = acquisition.publication_store
    deltas = [DisplayNavigationDelta(display.catalog_snapshot().entries[0])]
    deltas.extend(display.append_navigation("run.a", "/out/a.nxs", label)
                  for label in range(2, count + 1))
    return tuple(
        StandardRunEvent(
            identity,
            StandardEventKind.FRAME_READY,
            completed=index,
            total=count,
            artifact=delta.appended.artifact,
            frame_key=delta.appended,
            navigation_delta=delta,
            artifact_completed=index,
            artifact_total=count,
        )
        for index, delta in enumerate(deltas, start=1)
    )


def _batch_display(
    page: ScatteringWorkspace,
    executor: _Executor,
    identity: RunIdentity,
    *,
    configuration=None,
) -> tuple[DisplayNavigationDelta, ...]:
    from tests.xdart.scattering.test_e3_context_contract import (
        _acquisition, _view,
    )

    configuration = configuration or RunIntent(
        output_mode="Overwrite", processing_mode="Int 2D",
    ).freeze()
    _, acquisition = _acquisition(
        configuration=configuration, identity=identity,
    )
    executor.acquisition_context = (
        lambda candidate: acquisition if candidate is identity else None
    )
    controller = page._context_controller
    controller.adopt_acquisition(identity)
    display = acquisition.publication_store
    first = display.catalog_snapshot().entries[0]
    assert controller.select_navigation(first, (first,))
    page._refresh_shell()

    owner = display.artifacts[first.artifact]
    deltas = [DisplayNavigationDelta(first)]
    for label in (2, 3):
        view = _view(label, float(label))
        record = FrameRecord.from_view(view)
        publication = FramePublication(
            view,
            record=record,
            source_identity=f"{view.source_path}#{label}",
            scan_key=first.source_scan,
        )
        delta = display.append_navigation(
            first.source_scan, first.artifact, label,
        )
        display.retain_frame(
            owner,
            delta.appended,
            record,
            publication,
            source_identity=publication.source_identity,
            frame_mask_qualified=False,
        )
        display.put_payload(StandardDisplayPayload(
            0, delta.appended, f"Standard · run.a · frame {label}", view,
        ))
        deltas.append(delta)
    return tuple(deltas)


def test_performance_diagnostics_are_explicit_and_apply_next_run_values(
        qapp: QtWidgets.QApplication, monkeypatch) -> None:
    monkeypatch.setenv("XDART_PERF", "")
    monkeypatch.setenv("XDART_PERF_QUARTILES", "")
    monkeypatch.delenv("XDART_PERF", raising=False)
    monkeypatch.delenv("XDART_PERF_QUARTILES", raising=False)
    monkeypatch.delenv(
        "XDART_UNSAFE_UNFUNDED_STAGING_DIAGNOSTIC", raising=False,
    )
    executor = _Executor()
    page, _, _ = _active_page(executor)
    shell = _shell(page)
    try:
        config = shell.browser.findChild(
            QtWidgets.QToolButton, "configMenuButton",
        )
        assert config is not None
        assert "Performance Diagnostics…" in {
            action.text() for action in config.menu().actions()
        }
        dialog = PerformanceDiagnosticsDialog(page)
        quartile_checkbox = dialog.findChild(
            QtWidgets.QCheckBox, "performanceQuartileTelemetry",
        )
        assert quartile_checkbox is not None
        assert quartile_checkbox.text() == "Within-run quartile timing"
        dialog.deleteLater()
        initial = page._intents.snapshot()
        assert "_post_g2_pipeline_v2" not in initial.thaw().run_options
        assert "XDART_PERF" not in os.environ

        page._performance_diagnostics_editor = lambda *_args: (
            PerformanceDiagnosticsValues(
                1, 8, 16, 56, 375,
                save_xye=False,
                durable_fsync=False,
                staging_frame_cap=64,
                quartile_telemetry=True,
            )
        )
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.MENU, "Config:Performance Diagnostics…",
        ))

        applied = page._intents.snapshot()
        assert applied.revision == initial.revision + 1
        assert applied.thaw().run_options["_post_g2_pipeline_v2"] == {
            "writer_settlement_batch_size": 1,
            "nexus_record_batch_size": 8,
            "reduction_inflight": 16,
            "semantic_checkpoint_frame_cap": 56,
            "staging_frame_cap": 64,
        }
        assert applied.thaw().run_options[
            "_post_g2_output_diagnostics_v1"
        ] == {
            "save_xye": False,
            "durable_fsync": False,
        }
        assert page._live_plot_interval_ms == 375
        assert os.environ["XDART_PERF"] == "1"
        assert os.environ["XDART_PERF_QUARTILES"] == "1"
        assert "next run" in page._notice_text.lower()
        assert "crash" in page._notice_text.lower()

        monkeypatch.setenv(
            "XDART_UNSAFE_UNFUNDED_STAGING_DIAGNOSTIC", "1",
        )
        page._performance_diagnostics_editor = lambda *_args: (
            PerformanceDiagnosticsValues(
                1, 8, 8, 10_000, 375,
                staging_frame_cap=10_008,
            )
        )
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.MENU, "Config:Performance Diagnostics…",
        ))
        unsafe = page._intents.snapshot()
        assert unsafe.revision == applied.revision + 1
        assert unsafe.thaw().run_options[
            "_post_g2_unfunded_staging_diagnostic_v1"
        ] == {
            "mode": "UNSAFE_UNFUNDED",
            "checkpoint": 10_000,
            "staging_frame_cap": 10_008,
            "max_frames": 3_621,
        }
        assert "unsafe unfunded" in page._notice_text.lower()

        page._performance_diagnostics_editor = lambda *_args: (
            PerformanceDiagnosticsValues(
                1, 8, 8, 56, 375,
                staging_frame_cap=64,
                quartile_telemetry=False,
            )
        )
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.MENU, "Config:Performance Diagnostics…",
        ))
        ordinary = page._intents.snapshot()
        assert ordinary.revision == unsafe.revision + 1
        assert (
            "_post_g2_unfunded_staging_diagnostic_v1"
            not in ordinary.thaw().run_options
        )
        assert "XDART_PERF_QUARTILES" not in os.environ

        page._performance_diagnostics_editor = lambda *_args: None
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.MENU, "Config:Performance Diagnostics…",
        ))
        assert page._intents.snapshot().revision == ordinary.revision
        assert page._live_plot_interval_ms == 375
    finally:
        _dispose(page, qapp)


def test_performance_diagnostics_fresh_default_is_tuned_tuple(
        qapp: QtWidgets.QApplication, monkeypatch) -> None:
    monkeypatch.delenv("XDART_PERF_QUARTILES", raising=False)
    page, _, _ = _active_page(_Executor())
    dialog = PerformanceDiagnosticsDialog(page)
    monkeypatch.setattr(
        dialog,
        "exec",
        lambda: QtWidgets.QDialog.DialogCode.Accepted,
    )
    try:
        values = dialog.edit(page._intents.snapshot(), 375)
        assert values is not None
        assert (
            values.settlement,
            values.record,
            values.inflight,
            values.checkpoint,
            values.staging_frame_cap,
        ) == (1, 8, 8, 16, 64)
    finally:
        _dispose(page, qapp)


def test_xye_mode_drops_only_private_nexus_performance_options(
        qapp: QtWidgets.QApplication) -> None:
    nexus_only = {
        "_post_g2_pipeline_v2": {"pipeline": "stale"},
        "_post_g2_output_diagnostics_v1": {"save_xye": False},
        "_post_g2_unfunded_staging_diagnostic_v1": {"mode": "stale"},
    }
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=image_series_spec(Path("frame_0001.tif")),
            poni_file="calibration.poni",
            save_path="output.nxs",
            output_mode="Overwrite",
            run_options={**nexus_only, "unrelated": "preserved"},
        )),
        lifecycle=ScatteringCoordinator(),
        sources=_Sources(),
        executor=_Executor(),
    )
    try:
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SET_PROCESSING_MODE, "Int 1D (XYE)",
        ))

        intent = page._intents.snapshot().thaw()
        assert intent.processing_mode == "Int 1D (XYE)"
        assert intent.run_options == {"unrelated": "preserved"}
    finally:
        _dispose(page, qapp)


def test_xye_performance_dialog_applies_cadence_without_nexus_options(
        qapp: QtWidgets.QApplication, monkeypatch) -> None:
    monkeypatch.setenv("XDART_PERF", "")
    monkeypatch.setenv("XDART_PERF_QUARTILES", "")
    monkeypatch.delenv("XDART_PERF", raising=False)
    monkeypatch.delenv("XDART_PERF_QUARTILES", raising=False)
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=image_series_spec(Path("frame_0001.tif")),
            poni_file="calibration.poni",
            save_path="output.nxs",
            output_mode="Overwrite",
            processing_mode="Int 1D (XYE)",
            run_options={
                "_post_g2_pipeline_v2": {"pipeline": "stale"},
                "_post_g2_output_diagnostics_v1": {"save_xye": False},
                "unrelated": "preserved",
            },
        )),
        lifecycle=ScatteringCoordinator(),
        sources=_Sources(),
        executor=_Executor(),
    )
    page._performance_diagnostics_editor = lambda *_args: (
        PerformanceDiagnosticsValues(
            1, 8, 8, 16, 375,
            save_xye=False,
            durable_fsync=False,
            staging_frame_cap=64,
            quartile_telemetry=True,
        )
    )
    try:
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.MENU, "Config:Performance Diagnostics…",
        ))

        intent = page._intents.snapshot().thaw()
        assert intent.processing_mode == "Int 1D (XYE)"
        assert intent.run_options == {"unrelated": "preserved"}
        assert page._live_plot_interval_ms == 375
        assert os.environ["XDART_PERF_QUARTILES"] == "1"
        assert "plot cadence is active now" in page._notice_text
        assert "pipeline and output" not in page._notice_text
        assert "fsync" not in page._notice_text
    finally:
        _dispose(page, qapp)


def test_performance_diagnostics_reject_invalid_coupled_values(
        qapp: QtWidgets.QApplication, monkeypatch) -> None:
    monkeypatch.delenv("XDART_PERF", raising=False)
    page, _, _ = _active_page(_Executor())
    initial = page._intents.snapshot()
    try:
        page._performance_diagnostics_editor = lambda *_args: None
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.MENU, "Config:Performance Diagnostics…",
        ))
        assert page._intents.snapshot().revision == initial.revision
        assert "XDART_PERF" not in os.environ

        page._performance_diagnostics_editor = lambda *_args: (
            PerformanceDiagnosticsValues(5, 8, 4, 56, 125)
        )
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.MENU, "Config:Performance Diagnostics…",
        ))
        assert page._intents.snapshot().revision == initial.revision
        assert "batching bounds" in page._notice_text.lower()
        assert page._live_plot_interval_ms == 250
        assert "XDART_PERF" not in os.environ

        page._performance_diagnostics_editor = lambda *_args: (
            PerformanceDiagnosticsValues(
                1, 8, 8, 1_000, 125,
                staging_frame_cap=64,
            )
        )
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.MENU, "Config:Performance Diagnostics…",
        ))
        assert page._intents.snapshot().revision == initial.revision
        assert "staging" in page._notice_text.lower()
        assert "checkpoint" in page._notice_text.lower()
    finally:
        _dispose(page, qapp)


def test_terminal_elapsed_and_split_remain_in_run_status(
        qapp: QtWidgets.QApplication) -> None:
    executor = _Executor()
    page, _, identity = _active_page(executor)
    quartile_type = getattr(
        display_values_module, "StandardQuartileTiming",
    )
    quartiles = quartile_type(
        (163, 163, 163, 162),
        (
            ("wall", (5.0, 5.5, 5.75, 7.164)),
            ("reducer_compute", (4.0, 4.2, 4.4, 4.6)),
            ("source_read", (1.0, 1.1, 1.2, 1.95)),
            ("submit_wait", (2.5, 2.75, 3.0, 4.25)),
            ("writer_batch", (0.2, 0.3, 0.35, 0.4)),
            ("writer_flush", (0.1, 0.1, 0.1, 0.2)),
            ("xye", (0.4, 0.45, 0.5, 0.65)),
            ("completion_display", (0.15, 0.2, 0.2, 0.2)),
            ("finish_wait", (0.0, 0.0, 0.0, 3.0)),
        ),
        (160, 162, 163, 161),
    )
    timing = StandardTerminalTiming(
        23.414,
        20.0,
        3.414,
        (
            ("source_read", 5.25),
            ("submit_wait", 12.5),
            ("writer_batch", 1.25),
            ("writer_flush", 0.5),
            ("xye", 2.0),
            ("finish_wait", 3.0),
            ("display", 0.75),
        ),
        quartiles,
    )
    executor.events.append(StandardRunEvent(
        identity,
        StandardEventKind.FINISHED,
        completed=651,
        total=651,
        terminal_timing=timing,
    ))
    page._quartile_refresh_identity = identity
    page._quartile_refresh_seconds = [0.03, 0.04, 0.05, 0.06]
    try:
        page._drain_executor()
        label = _shell(page).run_controls.readinessLabel
        assert label.text() == "Complete · 651 Frames · 23.41 s"
        assert label.toolTip() == (
            "Total: 23.41 s\n"
            "Work: 20.00 s\n"
            "Cleanup: 3.41 s\n"
            "Source read: 5.25 s\n"
            "Submit/backpressure: 12.50 s\n"
            "NeXus write/checkpoint: 1.25 s\n"
            "Checkpoint flush/fsync: 0.50 s\n"
            "XYE: 2.00 s\n"
            "Finish/drain (includes terminal seal): 3.00 s\n"
            "Display: 0.75 s\n"
            "Parallel detail timers may overlap.\n"
            "Within-run quartiles (Q1 → Q4; Q4 includes terminal tail)\n"
            "Frames: 163 | 163 | 163 | 162\n"
            "Wall: 5.00 s | 5.50 s | 5.75 s | 7.16 s\n"
            "Reducer compute: 4.00 s / 160 (25.00 ms/frame) | "
            "4.20 s / 162 (25.93 ms/frame) | "
            "4.40 s / 163 (26.99 ms/frame) | "
            "4.60 s / 161 (28.57 ms/frame)\n"
            "Source read: 1.00 s | 1.10 s | 1.20 s | 1.95 s\n"
            "Submit/backpressure: 2.50 s | 2.75 s | 3.00 s | 4.25 s\n"
            "NeXus write/checkpoint: 0.20 s | 0.30 s | 0.35 s | 0.40 s\n"
            "Checkpoint flush/fsync: 0.10 s | 0.10 s | 0.10 s | 0.20 s\n"
            "XYE: 0.40 s | 0.45 s | 0.50 s | 0.65 s\n"
            "Completion/display: 0.15 s | 0.20 s | 0.20 s | 0.20 s\n"
            "Finish/drain: 0.00 s | 0.00 s | 0.00 s | 3.00 s\n"
            "GUI refresh (pre-terminal): 0.03 s | 0.04 s | 0.05 s | 0.06 s\n"
            "Quartile timers may overlap and do not sum to Wall.\n"
            "Submit/backpressure is orchestration wait, not pure integration."
        )
        page._refresh_shell()
        assert label.text() == "Complete · 651 Frames · 23.41 s"
    finally:
        _dispose(page, qapp)


def test_pending_failure_cannot_reset_or_launch(
    qapp: QtWidgets.QApplication,
) -> None:
    executor = _Executor()
    page, lifecycle, identity = _active_page(executor)
    executor.events.append(StandardRunEvent(identity, StandardEventKind.FAILED,
                                             cleanup_status=CleanupStatus.CLEANUP_PENDING))
    try:
        page._drain_executor()
        shell = _shell(page)
        shell.commandRequested.emit(
            ShellCommand(ShellCommandKind.RUN_ACTION)
        )
        page._drain_executor()
        responsive = []
        QtCore.QTimer.singleShot(0, lambda: responsive.append(True))
        qapp.processEvents()
        assert responsive == [True]
        assert lifecycle.phase is RunPhase.FAILED
        assert executor.start_calls == 0
        assert not shell.run_controls.startButton.isEnabled()
        assert (
            shell.run_controls.readinessLabel.text()
            == "Standard cleanup remains pending"
        )
    finally:
        _dispose(page, qapp)


def test_dense_bottom_deadline_keeps_real_images_separate_from_waterfall(
        qapp: QtWidgets.QApplication, monkeypatch) -> None:
    """The page, not the renderer, owns the latest dense-image deadline."""
    from tests.xdart.scattering.test_e3_context_contract import _view

    executor = _Executor()
    page, _, identity = _active_page(executor)
    page._preferences = replace(page._preferences, plot_mode="Waterfall")
    page._live_plot_interval_ms = 375
    events = _paced_frame_events(page, executor, identity, 5)
    acquisition = executor.acquisition_context(identity)
    assert acquisition is not None
    display = acquisition.publication_store
    display.max_payload_items = 16
    owner = display.artifacts["/out/a.nxs"]
    for event in events[1:]:
        frame = event.frame_key
        assert frame is not None
        view = _view(frame.local_frame_label, float(frame.local_frame_label))
        record = FrameRecord.from_view(view)
        publication = FramePublication(
            view,
            record=record,
            source_identity=f"{view.source_path}#{frame.local_frame_label}",
            scan_key=frame.source_scan,
        )
        display.retain_frame(
            owner,
            frame,
            record,
            publication,
            source_identity=publication.source_identity,
            frame_mask_qualified=False,
        )
        display.put_payload(StandardDisplayPayload(
            0, frame, f"Standard · run.a · frame {frame.local_frame_label}",
            view,
        ))
    now = [0.0]
    monkeypatch.setattr(
        page_module, "time", SimpleNamespace(monotonic=lambda: now[0]),
        raising=False,
    )
    try:
        executor.events.extend(events[:4])
        page._drain_executor()
        scientific = page._shell.scientific
        assert scientific._waterfall_y_values == (1.0, 2.0, 3.0, 4.0)
        rendered_projection = page._last_scientific_projection
        assert rendered_projection is not None
        assert scientific.bottom_waterfall_active
        assert not page._scientific_repaint_pending
        assert page._lifecycle.phase is RunPhase.RUNNING
        assert page._dense_plot_at == 0.0

        now[0] = 0.125
        executor.events.append(events[4])
        page._drain_executor()
        page._plot_deadline_timer.stop()
        assert page._waterfall_candidate_count == 5
        assert page._dense_plot_pending

        now[0] = 0.375
        page._drain_executor()
        assert page._dense_plot_pending
        assert page._live_plot_bottom_is_dense()
        assert scientific._waterfall_y_values == (1.0, 2.0, 3.0, 4.0)
        deadline = getattr(page, "_plot_deadline_timer", None)
        if deadline is not None:
            deadline.timeout.emit()
        assert scientific.raw.image.image[0, 0] == 5.0
        assert scientific.cake.image.image[0, 0] == 15.0
        assert scientific._waterfall_y_values == (1.0, 2.0, 3.0, 4.0)
        accepted = page._last_scientific_projection
        assert accepted is not None
        assert accepted.heavy is not None
        assert accepted.heavy.frame is events[4].frame_key
        assert accepted.title == rendered_projection.title
        assert accepted.traces is rendered_projection.traces

        now[0] = 0.5
        page._drain_executor()
        if deadline is not None:
            deadline.timeout.emit()
        assert scientific._waterfall_y_values == (1.0, 2.0, 3.0, 4.0, 5.0)
        assert page._last_scientific_projection is not None
        assert any(trace.frame is events[4].frame_key
                   for trace in page._last_scientific_projection.traces)

        now[0] = 0.55
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SET_PLOT_OPTION,
            2,
            path=("waterfall", "step"),
        ))
        assert scientific._waterfall_y_values == (1.0, 3.0, 5.0)
        assert not page._plot_deadline_timer.isActive()
    finally:
        _dispose(page, qapp)


def test_single_125ms_deadline_coalesces_and_direct_frame_bypasses(
        qapp: QtWidgets.QApplication, monkeypatch) -> None:
    """A new Single burst keeps its deadline after the target queue empties."""
    from tests.xdart.scattering.test_e3_context_contract import _view

    executor = _Executor()
    page, _, identity = _active_page(executor)
    page._live_plot_interval_ms = 125
    events = _paced_frame_events(page, executor, identity, 4)
    acquisition = executor.acquisition_context(identity)
    assert acquisition is not None
    display = acquisition.publication_store
    display.max_payload_items = 16
    owner = display.artifacts["/out/a.nxs"]
    for event in events[1:]:
        frame = event.frame_key
        assert frame is not None
        view = _view(frame.local_frame_label, float(frame.local_frame_label))
        record = FrameRecord.from_view(view)
        publication = FramePublication(
            view,
            record=record,
            source_identity=f"{view.source_path}#{frame.local_frame_label}",
            scan_key=frame.source_scan,
        )
        display.retain_frame(
            owner,
            frame,
            record,
            publication,
            source_identity=publication.source_identity,
            frame_mask_qualified=False,
        )
        display.put_payload(StandardDisplayPayload(
            0, frame, f"Standard · run.a · frame {frame.local_frame_label}",
            view,
        ))
    now = [0.0]
    monkeypatch.setattr(
        page_module, "time", SimpleNamespace(monotonic=lambda: now[0]),
        raising=False,
    )
    try:
        executor.events.append(events[0])
        page._drain_executor()
        scientific = page._shell.scientific
        assert scientific.raw.image.image[0, 0] == 1.0

        now[0] = 0.001
        executor.events.append(events[1])
        page._drain_executor()
        page._plot_deadline_timer.stop()
        assert page._image_plot_pending
        assert page._curve_plot_pending
        assert scientific.raw.image.image[0, 0] == 1.0

        now[0] = 0.05
        executor.events.append(events[2])
        page._drain_executor()
        assert scientific.raw.image.image[0, 0] == 1.0

        now[0] = 0.124
        page._plot_deadline_timer.timeout.emit()
        assert scientific.raw.image.image[0, 0] == 1.0

        now[0] = 0.125
        page._plot_deadline_timer.timeout.emit()
        assert scientific.raw.image.image[0, 0] == 3.0
        assert scientific.cake.image.image[0, 0] == 13.0
        assert scientific.trace_history_keys == (events[2].frame_key,)

        now[0] = 0.126
        executor.events.append(events[3])
        page._drain_executor()
        page._plot_deadline_timer.stop()
        assert page._live_plot_pending()
        frame = events[3].frame_key
        assert frame is not None
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SELECT_FRAME, frame=frame, frames=(frame,),
        ))
        assert scientific.raw.image.image[0, 0] == 4.0
        assert scientific.trace_history_keys == (frame,)
        assert not page._live_plot_pending()
        assert not page._plot_deadline_timer.isActive()
    finally:
        _dispose(page, qapp)


def test_plot_deadline_is_cancelled_at_terminal_and_close(
        qapp: QtWidgets.QApplication) -> None:
    executor = _Executor()
    page, _, identity = _active_page(executor)
    try:
        page._image_plot_at = time.monotonic()
        page._image_plot_pending = True
        page._arm_live_plot_deadline()
        assert page._plot_deadline_timer.isActive()

        executor.events.append(StandardRunEvent(
            identity,
            StandardEventKind.FINISHED,
            cleanup_status=CleanupStatus.CLEANED,
        ))
        page._drain_executor()
        assert not page._live_plot_pending()
        assert not page._plot_deadline_timer.isActive()

        page._image_plot_at = time.monotonic()
        page._image_plot_pending = True
        page._arm_live_plot_deadline()
        assert page._plot_deadline_timer.isActive()
        page.close_workspace()
        assert not page._live_plot_pending()
        assert not page._plot_deadline_timer.isActive()
    finally:
        _dispose(page, qapp)


def test_pacer_uses_exact_runtime_membership_without_scanning_navigation() -> None:
    identity = RunIdentity(7, "pacer-membership")
    owned = DisplayFrameKey(identity, "run.a", "/out/a.nxs", 1, 1)
    foreign_identity = RunIdentity(8, "pacer-membership-foreign")
    foreign = DisplayFrameKey(
        foreign_identity,
        "run.a",
        "/out/a.nxs",
        2,
        2,
    )

    class _HostileFrames:
        def __iter__(self):
            raise AssertionError("pacer scanned the immutable navigation prefix")

    exact_membership = {id(owned): owned}
    selected: list[tuple[DisplayFrameKey, tuple[DisplayFrameKey, ...]]] = []
    controller = SimpleNamespace(
        run_identity=identity,
        navigation=SimpleNamespace(frames=_HostileFrames()),
        owns_frame=lambda frame: exact_membership.get(id(frame)) is frame,
        select_navigation=lambda frame, frames: (
            selected.append((frame, frames)) or True
        ),
    )
    owner = SimpleNamespace(_context_controller=controller)

    assert ScatteringWorkspace._select_presentation_target(
        owner, identity, owned,
    )
    assert selected == [(owned, (owned,))]

    exact_membership.clear()
    assert ScatteringWorkspace._select_presentation_target(
        owner, identity, owned,
    ) is False
    assert ScatteringWorkspace._select_presentation_target(
        owner, foreign_identity, foreign,
    ) is False
    assert selected == [(owned, (owned,))]


def test_pacer_boundaries_clear_stale_work_and_flush_exact_latest(
        qapp: QtWidgets.QApplication, monkeypatch) -> None:
    executor = _Executor()
    page, lifecycle, identity = _active_page(executor)
    events = _paced_frame_events(page, executor, identity, 5)
    monkeypatch.setattr(page, "_follow_processed_artifact", lambda _frame: None)
    monkeypatch.setattr(page, "_refresh_shell", lambda **_kwargs: None)
    try:
        executor.events.extend(events)
        page._drain_executor()
        assert tuple(page._presentation_targets) == ()
        assert page._context_controller.navigation.current is events[-1].frame_key

        first = events[0].frame_key
        assert first is not None
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SELECT_FRAME,
            frame=first,
            frames=(first,),
        ))
        assert tuple(page._presentation_targets) == ()
        assert page._processed_browser.auto_last is False
        assert page._context_controller.navigation.current is first

        latest = events[-1].frame_key
        assert latest is not None

        def refill(*targets):
            page._presentation_targets.extend(targets or (latest,))
            page._presentation_run_identity = identity

        refill(*(
            event.frame_key for event in events[-3:]
            if event.frame_key is not None
        ))
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SET_AUTO_LAST, True,
        ))
        assert tuple(page._presentation_targets) == ()
        assert page._presentation_run_identity is None
        assert page._context_controller.navigation.current is latest

        refill()
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SET_PLOT_MODE, "Overlay",
        ))
        assert tuple(page._presentation_targets) == ()
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SET_PLOT_MODE, "Single",
        ))

        dispatched: list[tuple[str, int]] = []

        def pause():
            current = page._context_controller.navigation.current
            dispatched.append(("pause", current.local_frame_label))

        refill()
        monkeypatch.setattr(page._context_controller, "pause", pause)
        page._handle_shell_command(ShellCommand(ShellCommandKind.RUN_ACTION))
        assert dispatched == [("pause", 5)]
        assert tuple(page._presentation_targets) == ()

        def stop():
            current = page._context_controller.navigation.current
            dispatched.append(("stop", current.local_frame_label))

        refill()
        monkeypatch.setattr(page._context_controller, "stop", stop)
        page._handle_shell_command(ShellCommand(ShellCommandKind.STOP))
        assert dispatched[-1] == ("stop", 5)
        assert tuple(page._presentation_targets) == ()

        foreign_identity = RunIdentity(
            identity.generation + 1, f"{identity.fingerprint}-stale",
        )
        stale = DisplayFrameKey(
            foreign_identity, "run.a", "/out/a.nxs", 99, 99,
        )
        refill(stale, latest)
        assert page._context_controller.select_navigation(
            events[0].frame_key, (events[0].frame_key,),
        )
        page._drain_executor()
        assert page._context_controller.navigation.current is latest
        assert tuple(page._presentation_targets) == ()

        refill()
        executor.events.append(StandardRunEvent(
            identity, StandardEventKind.FINISHED,
            completed=5, total=5, artifact="/out/a.nxs",
            cleanup_status=CleanupStatus.CLEANED,
        ))
        page._drain_executor()
        assert page._context_controller.navigation.current is latest
        assert tuple(page._presentation_targets) == ()
        assert page._presentation_run_identity is None
        assert lifecycle.phase is RunPhase.IDLE
    finally:
        _dispose(page, qapp)


def test_executor_stop_failure_is_contained_at_qt_command_boundary(qapp: QtWidgets.QApplication) -> None:
    executor = _Executor()
    executor.stop_error = RuntimeError("stop dispatch failed")
    page, lifecycle, identity = _active_page(executor)
    try:
        shell = _shell(page)
        shell.commandRequested.emit(ShellCommand(ShellCommandKind.STOP))
        assert lifecycle.active_run_identity is identity
        assert lifecycle.phase is RunPhase.STOPPING
        assert "stop" in shell.scientific.status.text().lower()
    finally:
        executor.stop_error = None
        _dispose(page, qapp)


def test_executor_drain_timer_tracks_only_launched_run_lifetime(qapp: QtWidgets.QApplication,
                                                                 tmp_path: Path) -> None:
    executor = _Executor()
    source = image_series_spec(tmp_path / "frame_0001.tif")
    lifecycle = ScatteringCoordinator()
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(source_spec=source, poni_file=str(tmp_path / "calibration.poni"),
                                          save_path=str(tmp_path / "output.nxs"),
                                          output_mode="Overwrite")),
        lifecycle=lifecycle, sources=_Sources(), executor=executor,
    )
    try:
        assert lifecycle.phase is RunPhase.IDLE
        assert not page._run_timer.isActive()
        assert page._run_timer.interval() == 125
        shell = _shell(page)
        shell.commandRequested.emit(
            ShellCommand(ShellCommandKind.RUN_ACTION)
        )
        page._drain_executor()
        identity = lifecycle.active_run_identity
        assert identity is not None
        assert lifecycle.phase is RunPhase.RUNNING
        assert page._run_timer.isActive()
        executor.events.append(StandardRunEvent(identity, StandardEventKind.FINISHED,
                                                 artifact=str(tmp_path / "output.nxs"),
                                                 cleanup_status=CleanupStatus.CLEANED))
        page._drain_executor()
        assert lifecycle.phase is RunPhase.IDLE
        assert not page._run_timer.isActive()
        assert executor.start_calls == 1
    finally:
        _dispose(page, qapp)


def test_clean_nonbatch_terminal_starts_exact_browse_and_keeps_frame_labels(
    qapp: QtWidgets.QApplication,
    monkeypatch,
) -> None:
    """A completed Run authenticates its artifact without a second click."""

    from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest
    from xrd_tools.io.output_transaction import StreamTerminal

    executor = _Executor()
    frozen_root = "/frozen-project"
    page, lifecycle, identity = _active_page(
        executor, project_root=frozen_root
    )
    try:
        executor.events.extend(
            _paced_frame_events(
                page,
                executor,
                identity,
                3,
                project_root=frozen_root,
            )
        )
        page._drain_executor()
        navigation = page._context_controller.navigation
        assert navigation.current.local_frame_label == 3

        live_snapshot = page._intents.snapshot()
        edited = live_snapshot.thaw()
        edited.project_root = "/edited-after-admission"
        page._intents.commit(
            edited, expected_revision=live_snapshot.revision
        )

        seal = StreamTerminal(
            "/out/a.nxs", 1024, "d" * 64, 1, 1, 2, 3, 4,
        )
        request = BrowseLoadRequest(
            "terminal", 1, "/out/a.nxs", seal,
            source_root=frozen_root,
        )
        calls = []
        monkeypatch.setattr(
            page._context_controller,
            "begin_browse",
            lambda artifact, *, terminal_commit_identity=None, source_root=None:
                calls.append((artifact, terminal_commit_identity, source_root))
                or request,
        )
        paints = []
        monkeypatch.setattr(
            page, "_refresh_event_shell",
            lambda *, preserve_scientific=False,
            skip_scientific_projection=False:
                paints.append((
                    preserve_scientific,
                    skip_scientific_projection,
                )),
        )
        terminal = StandardRunEvent(
            identity,
            StandardEventKind.FINISHED,
            completed=3,
            total=3,
            artifact="/out/a.nxs",
            cleanup_status=CleanupStatus.CLEANED,
            terminal_commit_identity=seal,
        )
        executor.events.append(terminal)
        page._drain_executor()

        assert lifecycle.phase is RunPhase.IDLE
        assert calls == [("/out/a.nxs", seal, frozen_root)]
        assert page._processed_browser.terminal_handoff == TerminalBrowseHandoff(
            request, identity, "/out/a.nxs", 3, (3,), seal,
        )
        assert paints == [(True, True)]

        # A Stop without durable rows for its current artifact must not load
        # either an absent output or a restored prior file at the same name.
        page._begin_terminal_browse(
            replace(terminal, kind=StandardEventKind.STOPPED),
            was_batch=False,
            current=navigation.current,
            selected=navigation.selected,
        )
        page._begin_terminal_browse(
            replace(
                terminal, kind=StandardEventKind.STOPPED,
                artifact_completed=3, artifact_total=3,
            ),
            was_batch=False,
            current=navigation.current,
            selected=navigation.selected,
        )
        page._begin_terminal_browse(
            replace(terminal, cleanup_status=CleanupStatus.CLEANUP_PENDING),
            was_batch=False,
            current=navigation.current,
            selected=navigation.selected,
        )
        page._begin_terminal_browse(
            replace(
                terminal, kind=StandardEventKind.STOPPED,
                cleanup_status=CleanupStatus.CLEANUP_PENDING,
                artifact_completed=3, artifact_total=3,
                artifacts=(terminal.artifact,),
            ),
            was_batch=False,
            current=navigation.current,
            selected=navigation.selected,
        )
        page._begin_terminal_browse(
            terminal,
            was_batch=True,
            current=navigation.current,
            selected=navigation.selected,
        )
        assert calls == [("/out/a.nxs", seal, frozen_root)]
        assert page._processed_browser.terminal_handoff == TerminalBrowseHandoff(
            request, identity, "/out/a.nxs", 3, (3,), seal,
        )
    finally:
        _dispose(page, qapp)


def test_terminal_frame_signatures_are_zero_io_and_refuse_unknown_spelling(
    monkeypatch,
) -> None:
    """A 651-frame GUI rebind is lexical and refuses unknown aliases."""

    identity = RunIdentity.from_configuration(RunIntent().freeze())
    frames = tuple(
        DisplayFrameKey(
            identity,
            "scan",
            "/out/a.nxs" if index % 2 else "/alias/a.nxs",
            index,
            index,
        )
        for index in range(1, 652)
    )
    monkeypatch.setattr(
        page_module.os.path,
        "realpath",
        lambda _value: pytest.fail("terminal GUI rebind performed path I/O"),
    )
    canonical_by_artifact = {
        "/out/a.nxs": "/canonical/a.nxs",
        "/alias/a.nxs": "/canonical/a.nxs",
    }
    signatures = tuple(
        page_module._terminal_frame_signature(
            frame, canonical_by_artifact,
        )
        for frame in frames
    )

    assert len(signatures) == 651
    assert all(signature is not None for signature in signatures)
    unknown = DisplayFrameKey(
        identity, "scan", "/unknown/a.nxs", 652, 652,
    )
    assert page_module._terminal_frame_signature(
        unknown, canonical_by_artifact,
    ) is None


def test_terminal_browse_ready_performs_one_normal_scientific_reconcile(
    qapp: QtWidgets.QApplication,
    monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadOutcome, BrowseLoadRequest, BrowseLoadStatus,
    )

    executor = _Executor()
    page, _, identity = _active_page(executor)
    controller = page._context_controller
    request = BrowseLoadRequest("terminal-ready", 1, "/out/a.nxs")
    page._processed_browser.begin_terminal_handoff(
        request, identity, request.source_path, None, (), None,
        timing_start=None,
    )
    pending = [True]
    monkeypatch.setattr(
        type(controller), "browse_pending",
        property(lambda _self: pending[0]),
    )
    monkeypatch.setattr(
        controller, "poll_browse",
        lambda: pending.__setitem__(0, False) or BrowseLoadOutcome(
            request, BrowseLoadStatus.READY,
        ),
    )
    paints = []
    monkeypatch.setattr(
        page, "_refresh_event_shell",
        lambda *, preserve_scientific=False, **_kwargs:
            paints.append(preserve_scientific),
    )
    try:
        page._drain_executor()
        assert page._processed_browser.terminal_handoff is None
        assert paints == [False]
    finally:
        _dispose(page, qapp)


def test_terminal_browse_refusal_retires_exact_handoff_without_outcome(
    qapp: QtWidgets.QApplication,
    monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest

    executor = _Executor()
    page, _, identity = _active_page(executor)
    controller = page._context_controller
    request = BrowseLoadRequest("terminal-refused", 1, "/out/a.nxs")
    page._processed_browser.begin_terminal_handoff(
        request, identity, request.source_path, None, (), None,
        timing_start=None,
    )
    pending = [True]
    monkeypatch.setattr(
        type(controller), "browse_pending",
        property(lambda _self: pending[0]),
    )
    monkeypatch.setattr(
        controller, "poll_browse",
        lambda: pending.__setitem__(0, False) or None,
    )
    monkeypatch.setattr(
        controller, "owns_browse_request", lambda _request: False,
    )
    try:
        page._drain_executor()
        assert page._processed_browser.terminal_handoff is None
    finally:
        _dispose(page, qapp)


@pytest.mark.parametrize("kind", (StandardEventKind.FINISHED, StandardEventKind.STOPPED))
def test_clean_xye_terminal_does_not_browse_unwritten_nexus_target(
    qapp: QtWidgets.QApplication,
    monkeypatch,
    kind,
) -> None:
    """XYE-only publishes sidecars, so its planned NeXus path is not Browseable."""

    executor = _Executor()
    page, lifecycle, identity = _active_page(
        executor,
        processing_mode="Int 1D (XYE)",
    )
    try:
        executor.events.extend(_paced_frame_events(
            page,
            executor,
            identity,
            1,
            processing_mode="Int 1D (XYE)",
        ))
        page._drain_executor()
        browse_calls = []
        monkeypatch.setattr(
            page._context_controller,
            "begin_browse",
            lambda artifact: browse_calls.append(artifact),
        )
        executor.events.append(StandardRunEvent(
            identity,
            kind,
            completed=1,
            total=1,
            artifact="/out/planned-but-unwritten.nxs",
            cleanup_status=CleanupStatus.CLEANED,
            artifact_completed=1,
            artifact_total=1,
            artifacts=("/out/planned-but-unwritten.nxs",),
        ))
        page._drain_executor()

        assert lifecycle.phase is RunPhase.IDLE
        assert browse_calls == []
        assert page._processed_browser.terminal_handoff is None
        assert not page._context_controller.browse_pending
    finally:
        _dispose(page, qapp)


def test_terminal_browse_marker_survives_catalog_actions_and_clears_on_new_load(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Catalog-only actions do not orphan an active terminal handoff."""

    from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest

    executor = _Executor()
    page, _, identity = _active_page(executor)
    terminal = BrowseLoadRequest("terminal", 1, "/out/a.nxs")
    replacement = BrowseLoadRequest("replacement", 2, "/out/b.nxs")
    page._processed_browser.begin_terminal_handoff(
        terminal, identity, terminal.source_path, 5, (5,), None,
        timing_start=None,
    )
    directory_calls = []
    selected_targets = []
    browse_calls = []
    monkeypatch.setattr(
        page,
        "_set_browser_directory",
        lambda value, *, explicit: directory_calls.append((value, explicit)),
    )
    monkeypatch.setattr(
        page._context_controller,
        "select_browser_target",
        lambda value: selected_targets.append(value) or True,
    )
    monkeypatch.setattr(
        page._context_controller,
        "begin_browse",
        lambda value, *, source_root=None:
            browse_calls.append(value) or replacement,
    )
    monkeypatch.setattr(page, "_refresh_shell", lambda **_kwargs: None)
    forbidden_probe = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("browser activation performed GUI-thread filesystem I/O")
    )
    monkeypatch.setattr(page_module.os.path, "isdir", forbidden_probe)
    monkeypatch.setattr(Path, "resolve", forbidden_probe)
    try:
        page._select_scan(str(tmp_path), is_directory=True)
        assert directory_calls == [(str(tmp_path), True)]
        assert page._processed_browser.terminal_handoff.request is terminal

        page._select_scan(terminal.source_path)
        assert selected_targets == [terminal.source_path]
        assert page._processed_browser.terminal_handoff.request is terminal

        page._select_scan(replacement.source_path)
        assert browse_calls == [replacement.source_path]
        assert page._processed_browser.terminal_handoff is None
    finally:
        _dispose(page, qapp)


def test_fresh_run_projects_output_checking_not_cleanup(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    executor = _Executor()
    lifecycle = ScatteringCoordinator()
    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                source_spec=image_series_spec(tmp_path / "frame_0001.tif"),
                poni_file=str(tmp_path / "calibration.poni"),
                save_path=str(tmp_path / "output.nxs"),
                output_mode="Overwrite",
            )
        ),
        lifecycle=lifecycle,
        sources=_Sources(),
        executor=executor,
    )
    try:
        shell = _shell(page)
        shell.commandRequested.emit(
            ShellCommand(ShellCommandKind.RUN_ACTION)
        )

        assert lifecycle.phase is RunPhase.PREPARING
        assert executor.start_calls == 0
        assert page._admission is not None
        assert not shell.run_controls.startButton.isEnabled()
        assert (
            shell.run_controls.readinessLabel.text()
            == "Checking output targets…"
        )
    finally:
        _dispose(page, qapp)


def test_run_click_preserves_outgoing_paint_until_a_frame_arrives(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch,
) -> None:
    executor = _Executor()
    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                source_spec=image_series_spec(tmp_path / "frame_0001.tif"),
                poni_file=str(tmp_path / "calibration.poni"),
                save_path=str(tmp_path / "output.nxs"),
                output_mode="Overwrite",
            )
        ),
        lifecycle=ScatteringCoordinator(),
        sources=_Sources(),
        executor=executor,
    )
    refreshes: list[bool] = []
    monkeypatch.setattr(
        page,
        "_refresh_shell",
        lambda *, preserve_display=False: refreshes.append(
            preserve_display
        ),
    )
    try:
        _shell(page).commandRequested.emit(
            ShellCommand(ShellCommandKind.RUN_ACTION)
        )
        assert refreshes == [True]

        page._drain_executor()

        assert executor.start_calls == 1
        assert page._lifecycle.phase is RunPhase.RUNNING
        assert refreshes == [True, True]
    finally:
        _dispose(page, qapp)


@pytest.mark.parametrize(
    "refresh_options",
    ({"preserve_scientific": True}, {"preserve_display": True}),
)
def test_explicit_paint_preservation_skips_scientific_projection(
    qapp: QtWidgets.QApplication,
    monkeypatch,
    refresh_options: dict[str, bool],
) -> None:
    executor = _Executor()
    page, _, identity = _active_page(executor)
    _paced_frame_events(page, executor, identity, 1)
    controller = page._context_controller
    page._refresh_shell()
    projection = page._last_scientific_projection
    assert projection is not None
    raw = _shell(page).scientific.raw.image.image
    cake = _shell(page).scientific.cake.image.image
    project_calls = []
    commit_calls = []
    monkeypatch.setattr(
        controller,
        "project_navigation",
        lambda **_options: project_calls.append(True) or (),
    )
    monkeypatch.setattr(
        controller,
        "commit_navigation_projection",
        lambda *_args, **_options: commit_calls.append(True),
    )
    try:
        page._refresh_shell(**refresh_options)

        assert project_calls == []
        assert commit_calls == []
        assert page._last_scientific_projection is projection
        assert _shell(page).scientific.raw.image.image is raw
        assert _shell(page).scientific.cake.image.image is cake
    finally:
        _dispose(page, qapp)


def test_directory_selection_updates_browser_without_clearing_science(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    executor = _Executor()
    page, _, identity = _active_page(executor)
    _paced_frame_events(page, executor, identity, 1)
    owner = page._processed_browser
    old_directory = tmp_path / "old"
    old_directory.mkdir()
    (old_directory / "out.nexus").touch()
    owner.set_directory(
        str(old_directory), explicit=False,
    )
    _wait_until(
        qapp,
        lambda: owner.active_request is None
        and owner.queued_request is None,
    )
    assert owner.catalog
    page._refresh_shell()
    shell = _shell(page)
    browser = shell.browser
    assert browser.scans.count() > 0
    old_directory_text = browser.directory_label.text()
    old_directory_path = browser.directory_label.toolTip()
    scientific = shell.scientific
    current = page._context_controller.navigation.current
    assert current is not None
    traces = scientific.curve.listDataItems()
    assert len(traces) == 1
    trace = traces[0]
    raw = scientific.raw.image.image
    cake = scientific.cake.image.image
    trace_x, trace_y = trace.xData, trace.yData
    bottom = scientific.bottom_stack.currentWidget()
    background_owner = page._background_owner
    expected_background = scientific._expected_background_key
    rendered_background = scientific._rendered_background_key
    page._retain_outgoing_display = True
    page._presentation_run_identity = identity
    page._presentation_targets.append(current)
    try:
        shell.commandRequested.emit(ShellCommand(
            ShellCommandKind.SELECT_SCAN,
            value=str(tmp_path),
            path=("directory",),
        ))

        assert owner.directory == str(tmp_path)
        assert owner.catalog == ()
        catalog_request = owner.active_request or owner.queued_request
        assert catalog_request is not None
        assert catalog_request.directory == str(tmp_path)
        assert browser.directory_label.toolTip() == str(tmp_path)
        assert browser.directory_label.toolTip() != old_directory_path
        assert browser.directory_label.text()
        assert browser.directory_label.text() != old_directory_text
        assert browser.scans.count() == 0
        assert page._presentation_run_identity is identity
        assert tuple(page._presentation_targets) == (current,)
        assert scientific.bottom_stack.currentWidget() is bottom
        assert scientific.raw.image.image is raw
        assert scientific.cake.image.image is cake
        assert scientific.curve.listDataItems()[0] is trace
        np.testing.assert_array_equal(trace.xData, trace_x)
        np.testing.assert_array_equal(trace.yData, trace_y)
        assert page._background_owner is background_owner
        assert scientific._expected_background_key is expected_background
        assert scientific._rendered_background_key is rendered_background
    finally:
        _dispose(page, qapp)


@pytest.mark.parametrize(
    ("event_index", "expected_selection_rebuilds"),
    ((0, 0), (1, 1)),
)
def test_single_first_paced_frame_skips_only_exact_selection_rebuild(
    qapp: QtWidgets.QApplication,
    monkeypatch,
    event_index: int,
    expected_selection_rebuilds: int,
) -> None:
    executor = _Executor()
    page, _, identity = _active_page(executor)
    page._preferences = replace(page._preferences, plot_mode="Single")
    events = _paced_frame_events(page, executor, identity, 2)
    controller = page._context_controller
    selection_rebuilds = []
    select_navigation = controller.select_navigation

    def select(current, selected):
        selection_rebuilds.append((current, selected))
        return select_navigation(current, selected)

    monkeypatch.setattr(controller, "select_navigation", select)
    monkeypatch.setattr(page, "_follow_processed_artifact", lambda _frame: None)
    monkeypatch.setattr(page, "_refresh_event_shell", lambda **_options: None)
    frame = events[event_index].navigation_delta.appended
    try:
        executor.events.append(events[event_index])
        page._drain_executor()

        assert len(selection_rebuilds) == expected_selection_rebuilds
        assert controller.navigation.current is frame
        assert controller.navigation.selected == (frame,)
        assert page._run_frame_seen is True
    finally:
        _dispose(page, qapp)


def test_post_retirement_start_refusal_preserves_detached_outgoing_paint(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch,
) -> None:
    class _StartRefusalExecutor(_Executor):
        def start(self, _configuration, _source, run_identity, _admission):
            self.start_calls += 1
            self.last_identity = run_identity
            return ExecutorStartFailed(run_identity, CleanupStatus.CLEANED)

    executor = _StartRefusalExecutor()
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=image_series_spec(tmp_path / "frame_0001.tif"),
            poni_file=str(tmp_path / "calibration.poni"),
            save_path=str(tmp_path / "output.nxs"),
            output_mode="Overwrite",
        )),
        lifecycle=ScatteringCoordinator(),
        sources=_Sources(),
        executor=executor,
    )
    refreshes: list[bool] = []
    retirements: list[DisplayRetirementReceipt] = []
    real_apply = page._context_controller.apply_display_retirement

    def applied(receipt):
        retirements.append(receipt)
        return real_apply(receipt)

    monkeypatch.setattr(
        page._context_controller, "apply_display_retirement", applied,
    )
    monkeypatch.setattr(
        page,
        "_refresh_shell",
        lambda *, preserve_display=False: refreshes.append(
            preserve_display or page._retain_outgoing_display
        ),
    )
    try:
        _shell(page).commandRequested.emit(
            ShellCommand(ShellCommandKind.RUN_ACTION)
        )
        page._drain_executor()

        assert executor.start_calls == 1
        assert len(retirements) == 1
        assert page._lifecycle.phase is RunPhase.FAILED
        assert page._admission is None
        assert page._retain_outgoing_display is True
        assert refreshes and all(refreshes)
    finally:
        _dispose(page, qapp)


def test_settled_browse_payload_cannot_release_new_run_paint_hold(
    qapp: QtWidgets.QApplication,
) -> None:
    """A resident outgoing Browse frame is not the new run's first paint."""

    from tests.xdart.scattering.test_e3_context_contract import (
        _acquisition,
        _browse,
    )
    from xdart.modules.display_context import ContextKind, new_context_token

    executor = _Executor()
    page, lifecycle, incoming_identity = _active_page(executor)
    controller = page._context_controller
    runtime = controller._runtime
    outgoing_identity = RunIdentity(
        incoming_identity.generation,
        f"{incoming_identity.fingerprint}-browse",
    )
    runtime._run_identity = outgoing_identity
    request, browse = _browse(
        new_context_token(ContextKind.BROWSE),
        1,
        scan_key="terminal",
    )
    runtime.adopt_browse(browse, request)
    shell = _shell(page)
    try:
        page._refresh_shell()
        outgoing = controller.navigation.current
        assert outgoing is not None
        assert outgoing.run_identity is outgoing_identity
        assert lifecycle.active_run_identity is incoming_identity
        outgoing_title = shell.scientific.title.text()
        outgoing_raw = np.array(
            shell.scientific.raw.image.image,
            copy=True,
        )

        page._retain_outgoing_display = True
        page._refresh_shell(preserve_display=True)
        assert page._retain_outgoing_display is True
        assert shell.scientific.title.text() == outgoing_title
        np.testing.assert_array_equal(
            shell.scientific.raw.image.image,
            outgoing_raw,
        )

        runtime.clear_browse(select_acquisition=False)
        configuration = RunIntent(
            output_mode="Overwrite",
            processing_mode="Int 2D",
        ).freeze()
        assert configuration.identity == (
            incoming_identity.generation,
            incoming_identity.fingerprint,
        )
        _, acquisition = _acquisition(
            configuration=configuration,
            identity=incoming_identity,
        )
        runtime.adopt_acquisition(incoming_identity, acquisition)
        page._refresh_shell()
        incoming = controller.navigation.current
        assert incoming is not None
        assert incoming.run_identity is incoming_identity
        assert controller.run_identity is incoming_identity
        assert page._retain_outgoing_display is False
        assert not np.array_equal(
            shell.scientific.raw.image.image,
            outgoing_raw,
        )
    finally:
        _dispose(page, qapp)


def test_historical_frame_disables_auto_last_and_reenable_selects_latest(
    qapp: QtWidgets.QApplication,
    monkeypatch,
) -> None:
    executor = _Executor()
    page, _, identity = _active_page(executor)
    first = DisplayFrameKey(identity, "scan", "output.nxs", 1, 1)
    latest = DisplayFrameKey(identity, "scan", "output.nxs", 2, 2)
    runtime = page._context_controller._runtime
    runtime._acquisition_navigation = FrameNavigationProjection(
        (first, latest),
        latest,
        (latest,),
    )

    def select_navigation(frame, frames):
        runtime._acquisition_navigation = FrameNavigationProjection(
            (first, latest),
            frame,
            frames,
        )
        return True

    selected_plot_modes: list[str] = []

    def select_latest_navigation(*, plot_mode="Single"):
        selected_plot_modes.append(plot_mode)
        runtime._acquisition_navigation = FrameNavigationProjection(
            (first, latest),
            latest,
            (latest,),
        )
        return True

    monkeypatch.setattr(
        page._context_controller,
        "owns_frame",
        lambda frame: frame in {first, latest},
    )
    monkeypatch.setattr(
        page._context_controller,
        "select_navigation",
        select_navigation,
    )
    monkeypatch.setattr(
        page._context_controller,
        "select_latest_navigation",
        select_latest_navigation,
    )
    try:
        page._handle_shell_command(
            ShellCommand(
                ShellCommandKind.SELECT_FRAME,
                frame=first,
                frames=(first,),
            )
        )
        assert page._processed_browser.auto_last is False
        assert page._context_controller.navigation.current is first

        page._handle_shell_command(
            ShellCommand(ShellCommandKind.SET_AUTO_LAST, True)
        )
        assert page._processed_browser.auto_last is True
        assert selected_plot_modes == ["Single"]
        assert page._context_controller.navigation.current is latest
        assert page._context_controller.navigation.selected == (latest,)
    finally:
        _dispose(page, qapp)


def test_cold_frame_refresh_catches_up_distinct_acquisition_owner_before_paint(
    qapp: QtWidgets.QApplication,
    monkeypatch,
) -> None:
    from tests.xdart.scattering.test_e3_context_contract import _acquisition

    executor = _Executor()
    page, _, identity = _active_page(executor)
    configuration = RunIntent(output_mode="Overwrite").freeze()
    _, acquisition = _acquisition(
        configuration=configuration,
        identity=identity,
    )
    executor.acquisition_context = lambda candidate: (
        acquisition if candidate is identity else None
    )
    controller = page._context_controller
    controller.adopt_acquisition(identity)
    first = controller.navigation.current
    assert first is not None
    acquisition.rescope_to("run.b", "/data/b_0001.tif")
    assert controller.project_navigation() == ()
    page._retain_outgoing_display = True
    monkeypatch.setattr(page, "_follow_processed_artifact", lambda _frame: None)

    try:
        executor.events.append(
            StandardRunEvent(
                identity,
                StandardEventKind.FRAME_READY,
                completed=1,
                total=2,
                artifact=first.artifact,
                frame_key=first,
                navigation_delta=DisplayNavigationDelta(first),
            )
        )
        page._drain_executor()

        selection = controller.selection
        assert selection is not None
        assert selection.owner == acquisition.hydration_owner
        assert controller.navigation.current is first
        assert page._retain_outgoing_display is False
        shell = _shell(page)
        assert shell.scientific.title.text() != "Current"
        assert shell.scientific.raw.image.image is not None
        assert shell.scientific.cake.image.image is not None
        assert shell.scientific.curve.listDataItems()
    finally:
        _dispose(page, qapp)


def test_noop_append_display_ready_replaces_outgoing_paint_while_running(
    qapp: QtWidgets.QApplication,
    monkeypatch,
) -> None:
    executor = _Executor()
    page, lifecycle, identity = _active_page(
        executor, output_mode="Append",
    )
    frames = tuple(
        DisplayFrameKey(
            identity, "run.append", "/out/noop.nxs", label, label,
        )
        for label in range(1, 6)
    )
    runtime = page._context_controller._runtime
    runtime._run_identity = identity
    runtime._acquisition_navigation = FrameNavigationProjection(
        frames, frames[-1], (frames[-1],),
    )
    payloads: dict[DisplayFrameKey, StandardDisplayPayload] = {}
    monkeypatch.setattr(
        runtime,
        "resident_frame_keys",
        lambda *_args: frozenset(payloads),
    )
    monkeypatch.setattr(
        page._context_controller,
        "project_navigation",
        lambda **_kwargs: tuple(
            payloads[frame]
            for frame in runtime.navigation.selected
            if frame in payloads
        ),
    )
    monkeypatch.setattr(
        page._context_controller,
        "owns_frame",
        lambda frame: any(frame is candidate for candidate in frames),
    )

    def select_navigation(frame, selected):
        runtime._acquisition_navigation = FrameNavigationProjection(
            frames, frame, selected,
        )
        return True

    monkeypatch.setattr(
        page._context_controller, "select_navigation", select_navigation,
    )

    def qualify(event):
        if (
            event.run_identity is identity
            and any(event.frame_key is frame for frame in frames)
            and event.selection_generation == 0
        ):
            return payloads.get(event.frame_key)
        return None

    monkeypatch.setattr(
        page._context_controller, "qualify_display_event", qualify,
    )
    shell = _shell(page)
    page._retain_outgoing_display = True
    events: list[StandardRunEvent] = []

    def install_hydrated(label: int) -> DisplayFrameKey:
        key = frames[label - 1]
        raw = np.full((2, 3), float(label))
        view = FrameView(
            label,
            raw=raw,
            thumbnail=raw,
            source_path=f"/data/scan_{label}.tif",
            source_frame_index=label,
        )
        payloads[key] = StandardDisplayPayload(
            0, key, f"frame {label}", view,
        )
        event = StandardRunEvent(
            identity,
            StandardEventKind.DISPLAY_READY,
            artifact=key.artifact,
            frame_key=key,
            selection_generation=0,
        )
        events.append(event)
        executor.events.append(event)
        return key

    try:
        page._refresh_shell(preserve_display=True)
        controller = page._context_controller
        assert tuple(
            frame.local_frame_label for frame in controller.navigation.frames
        ) == (1, 2, 3, 4, 5)
        assert controller.navigation.current is frames[-1]
        assert page._retain_outgoing_display is True
        assert page._run_frame_seen is False

        foreign = (
            StandardRunEvent(
                RunIdentity(
                    identity.generation + 1,
                    f"{identity.fingerprint}-stale",
                ),
                StandardEventKind.DISPLAY_READY,
                artifact=frames[-1].artifact,
                selection_generation=0,
            ),
            StandardRunEvent(
                identity,
                StandardEventKind.DISPLAY_READY,
                artifact="foreign.nxs",
                frame_key=DisplayFrameKey(
                    identity, "foreign", "foreign.nxs", 1, 1,
                ),
                selection_generation=0,
            ),
        )
        events.extend(foreign)
        executor.events.extend(foreign)
        page._drain_executor()
        assert page._retain_outgoing_display is True

        page._batch_terminal.begin_run(
            identity, batch_mode=True, visible_progress=page._progress,
        )
        latest = install_hydrated(5)
        page._drain_executor()
        assert page._retain_outgoing_display is True

        page._batch_terminal.retire(force=True)
        executor.events.append(events[-1])
        page._drain_executor()
        assert lifecycle.phase is RunPhase.RUNNING
        assert page._run_frame_seen is False
        assert page._retain_outgoing_display is False
        assert not any(
            event.kind is StandardEventKind.FRAME_READY for event in events
        )
        assert shell.scientific.frame_selector.currentData() is latest
        assert shell.scientific.progress.text() == "5/5"
        assert shell.scientific.title.text() == "scan_5.tif"

        first = frames[0]
        shell.scientific.frame_selector.setCurrentIndex(0)
        assert controller.navigation.current is first
        install_hydrated(1)
        page._drain_executor()

        assert lifecycle.phase is RunPhase.RUNNING
        assert page._run_frame_seen is False
        assert not any(
            event.kind is StandardEventKind.FRAME_READY for event in events
        )
        assert shell.scientific.frame_selector.currentData() is first
        assert shell.scientific.progress.text() == "1/5"
        assert shell.scientific.title.text() == "scan_1.tif"
        assert shell.scientific.raw.image.image is not None
    finally:
        _dispose(page, qapp)


def test_real_batch_click_freezes_active_batch_and_defers_all_frame_science(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch,
) -> None:
    executor = _Executor()
    lifecycle = ScatteringCoordinator()
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=image_series_spec(tmp_path / "frame_0001.tif"),
            poni_file=str(tmp_path / "calibration.poni"),
            save_path=str(tmp_path / "output.nxs"),
            output_mode="Overwrite",
            processing_mode="Int 2D",
        )),
        lifecycle=lifecycle,
        sources=_Sources(),
        executor=executor,
    )
    try:
        shell = _shell(page)
        assert shell.run_controls.startButton.isEnabled()
        assert not shell.run_controls.batchButton.isChecked()
        shell.run_controls.batchButton.click()
        assert page._intents.snapshot().thaw().batch_mode
        shell.run_controls.startButton.click()
        page._drain_executor()

        identity = lifecycle.active_run_identity
        assert identity is not None
        assert executor.start_calls == 1
        assert executor.last_configuration is not None
        assert executor.last_configuration.batch_mode
        assert page._batch_terminal.active
        assert page._batch_terminal.presentation is None
        assert page._retain_outgoing_display

        deltas = _batch_display(
            page,
            executor,
            identity,
            configuration=executor.last_configuration,
        )
        controller = page._context_controller
        scientific = shell.scientific
        visible_frames = shell.browser.frame_model.frames
        visible_current = shell.browser._committed_current
        visible_status = shell.run_controls.readinessLabel.full_text()
        visible_transient = page._processed_browser.transient_frame
        calls = {
            "project": 0,
            "qualify": 0,
            "commit": 0,
            "reconcile": 0,
            "sync_detector": 0,
            "request_full": 0,
            "traces": 0,
            "raw": 0,
            "cake": 0,
            "waterfall": 0,
        }
        real_project = controller.project_navigation
        real_qualify = controller.qualify_display_event
        real_commit = controller.commit_navigation_projection
        real_reconcile = scientific.reconcile
        real_sync_detector = page._sync_detector_demand
        real_request_full = controller.request_full_current
        real_traces = scientific._render_traces
        real_raw = scientific.raw.render
        real_cake = scientific.cake.render
        real_waterfall = scientific.waterfall.render

        def project(**options):
            calls["project"] += 1
            return real_project(**options)

        def qualify(event):
            calls["qualify"] += 1
            return real_qualify(event)

        def commit(frames):
            calls["commit"] += 1
            return real_commit(frames)

        def reconcile(*args, **options):
            calls["reconcile"] += 1
            return real_reconcile(*args, **options)

        def sync_detector():
            calls["sync_detector"] += 1
            return real_sync_detector()

        def request_full():
            calls["request_full"] += 1
            return real_request_full()

        def render_traces(*args, **options):
            calls["traces"] += 1
            return real_traces(*args, **options)

        def render_raw(*args, **options):
            calls["raw"] += 1
            return real_raw(*args, **options)

        def render_cake(*args, **options):
            calls["cake"] += 1
            return real_cake(*args, **options)

        def render_waterfall(*args, **options):
            calls["waterfall"] += 1
            return real_waterfall(*args, **options)

        monkeypatch.setattr(controller, "project_navigation", project)
        monkeypatch.setattr(controller, "qualify_display_event", qualify)
        monkeypatch.setattr(controller, "commit_navigation_projection", commit)
        monkeypatch.setattr(scientific, "reconcile", reconcile)
        monkeypatch.setattr(page, "_sync_detector_demand", sync_detector)
        monkeypatch.setattr(controller, "request_full_current", request_full)
        monkeypatch.setattr(scientific, "_render_traces", render_traces)
        monkeypatch.setattr(scientific.raw, "render", render_raw)
        monkeypatch.setattr(scientific.cake, "render", render_cake)
        monkeypatch.setattr(scientific.waterfall, "render", render_waterfall)

        for completed, delta in enumerate(deltas, start=1):
            executor.events.append(StandardRunEvent(
                identity,
                StandardEventKind.FRAME_READY,
                completed=completed,
                total=len(deltas),
                artifact=delta.appended.artifact,
                detail=f"Batch {completed}/{len(deltas)}",
                frame_key=delta.appended,
                navigation_delta=delta,
            ))
            page._drain_executor()
            assert page._batch_terminal.active
            assert page._batch_terminal.latest_frame is delta.appended
            assert page._progress.completed == completed
            assert page._progress.total == len(deltas)
            assert shell.browser.frame_model.frames == visible_frames
            assert shell.browser._committed_current is visible_current
            assert shell.run_controls.readinessLabel.full_text() == (
                visible_status
            )
            assert page._processed_browser.transient_frame is visible_transient
            if completed == 1:
                page._handle_shell_command(ShellCommand(
                    ShellCommandKind.SET_CORES, 2,
                ))
                assert page._batch_terminal.active
                assert page._batch_terminal.latest_frame is delta.appended
                assert shell.browser.frame_model.frames == visible_frames
                assert shell.browser._committed_current is visible_current
                assert shell.run_controls.readinessLabel.full_text() == (
                    visible_status
                )
                assert page._processed_browser.transient_frame is visible_transient

        assert calls == {
            "project": 0,
            "qualify": 0,
            "commit": 0,
            "reconcile": 0,
            "sync_detector": 0,
            "request_full": 0,
            "traces": 0,
            "raw": 0,
            "cake": 0,
            "waterfall": 0,
        }
        assert page._batch_terminal.presentation is None
        assert lifecycle.phase is RunPhase.RUNNING
    finally:
        _dispose(page, qapp)


@pytest.mark.parametrize(
    ("plot_mode", "detector_mode"),
    (
        ("Single", "thumbnail"),
        ("Overlay", "thumbnail"),
        ("Waterfall", "full"),
    ),
)
def test_batch_run_projects_only_exact_latest_frame_at_terminal(
    qapp: QtWidgets.QApplication,
    monkeypatch,
    plot_mode: str,
    detector_mode: str,
) -> None:
    executor = _Executor()
    page, lifecycle, identity = _active_page(executor)
    try:
        page._browser_catalog_timer.stop()
        retired_browser = page._processed_browser
        retired_browser.begin_close()
        _wait_until(qapp, retired_browser.retry_close)
        catalog_gates = (Event(), Event())
        catalog_calls: list[str] = []
        catalog_wakes = []

        def read_catalog(directory, **_kwargs):
            index = len(catalog_calls)
            assert index < len(catalog_gates)
            catalog_calls.append(directory)
            assert catalog_gates[index].wait(5.0)
            return ()

        browser_owner = ProcessedBrowserOwner(
            save_path="/out/output.nexus",
            processing_mode="Int 2D",
            deliver=catalog_wakes.append,
            catalog_reader=read_catalog,
        )
        page._processed_browser = browser_owner
        browser_owner.begin_follow(identity)
        deltas = _batch_display(page, executor, identity)
        assert browser_owner.active_request is None
        assert browser_owner.queued_request is None
        controller = page._context_controller
        shell = _shell(page)
        scientific = shell.scientific
        browser = shell.browser
        old_projection = page._last_scientific_projection
        old_raw = scientific.raw.image.image
        old_cake = scientific.cake.image.image
        old_items = tuple(scientific.curve.listDataItems())
        assert old_projection is not None
        assert old_raw is not None and old_cake is not None
        assert len(old_items) == 1
        old_item = old_items[0]
        old_x, old_y = old_item.xData, old_item.yData

        calls = {
            "project": 0,
            "commit": 0,
            "reconcile": 0,
            "traces": 0,
            "raw": 0,
            "cake": 0,
            "waterfall": 0,
            "sync_detector": 0,
            "clear_full": 0,
            "request_full": 0,
        }
        real_sync_detector = page._sync_detector_demand
        real_project = controller.project_navigation
        real_commit = controller.commit_navigation_projection
        real_reconcile = scientific.reconcile
        real_traces = scientific._render_traces
        real_raw = scientific.raw.render
        real_cake = scientific.cake.render
        real_waterfall = scientific.waterfall.render

        def project(**options):
            calls["project"] += 1
            return real_project(**options)

        def sync_detector():
            calls["sync_detector"] += 1
            return real_sync_detector()

        def commit(frames):
            calls["commit"] += 1
            return real_commit(frames)

        def reconcile(*args, **options):
            calls["reconcile"] += 1
            return real_reconcile(*args, **options)

        def render_traces(*args, **options):
            calls["traces"] += 1
            return real_traces(*args, **options)

        def render_raw(*args, **options):
            calls["raw"] += 1
            return real_raw(*args, **options)

        def render_cake(*args, **options):
            calls["cake"] += 1
            return real_cake(*args, **options)

        def render_waterfall(*args, **options):
            calls["waterfall"] += 1
            return real_waterfall(*args, **options)

        def clear_full():
            calls["clear_full"] += 1
            return True

        def request_full():
            calls["request_full"] += 1
            return None

        monkeypatch.setattr(controller, "project_navigation", project)
        monkeypatch.setattr(
            page, "_sync_detector_demand", sync_detector,
        )
        monkeypatch.setattr(
            controller, "commit_navigation_projection", commit,
        )
        monkeypatch.setattr(scientific, "reconcile", reconcile)
        monkeypatch.setattr(scientific, "_render_traces", render_traces)
        monkeypatch.setattr(scientific.raw, "render", render_raw)
        monkeypatch.setattr(scientific.cake, "render", render_cake)
        monkeypatch.setattr(scientific.waterfall, "render", render_waterfall)
        monkeypatch.setattr(controller, "clear_full_raw", clear_full)
        monkeypatch.setattr(
            controller, "request_full_current", request_full,
        )

        page._preferences = replace(
            page._preferences,
            plot_mode=plot_mode,
            detector_mode=detector_mode,
        )
        selection = controller.selection
        assert selection is not None
        assert browser_owner.directory == os.path.dirname(
            deltas[0].appended.artifact
        )
        page._batch_terminal.begin_run(
            identity, batch_mode=True, visible_progress=page._progress,
        )
        page._retain_outgoing_display = True
        visible_frames = browser.frame_model.frames
        visible_current = browser._committed_current
        visible_status = shell.run_controls.readinessLabel.full_text()
        visible_transient = page._processed_browser.transient_frame
        for completed, delta in enumerate(deltas, start=1):
            frame = delta.appended
            executor.events.append(StandardRunEvent(
                identity,
                StandardEventKind.FRAME_READY,
                completed=completed,
                total=3,
                artifact=frame.artifact,
                detail=f"Batch {completed}/3",
                frame_key=frame,
                navigation_delta=delta,
            ))
            page._drain_executor()
            assert page._progress.completed == completed
            assert page._progress.total == 3
            assert browser.frame_model.frames == visible_frames
            assert browser._committed_current is visible_current
            assert shell.run_controls.readinessLabel.full_text() == (
                visible_status
            )
            assert page._processed_browser.transient_frame is visible_transient

        # Exercise an otherwise ordinary control refresh and the acquisition
        # rescope catch-up branch without replacing the production refresh.
        with monkeypatch.context() as scoped:
            synchronized = []
            scoped.setattr(
                controller,
                "synchronize_acquisition_scope",
                lambda: synchronized.append(True) or True,
            )
            page._handle_shell_command(ShellCommand(
                ShellCommandKind.SET_CORES, 2,
            ))
            assert synchronized == [True]
        assert page._intents.snapshot().thaw().max_cores == 2
        assert browser.frame_model.frames == visible_frames
        assert browser._committed_current is visible_current
        assert shell.run_controls.readinessLabel.full_text() == visible_status
        assert page._processed_browser.transient_frame is visible_transient

        assert calls == {
            "project": 0,
            "commit": 0,
            "reconcile": 0,
            "traces": 0,
            "raw": 0,
            "cake": 0,
            "waterfall": 0,
            "sync_detector": 0,
            "clear_full": 0,
            "request_full": 0,
        }
        assert page._last_scientific_projection is old_projection
        assert scientific.raw.image.image is old_raw
        assert scientific.cake.image.image is old_cake
        assert tuple(scientific.curve.listDataItems()) == (old_item,)
        assert old_item.xData is old_x and old_item.yData is old_y
        assert page._retain_outgoing_display is True
        assert lifecycle.phase is RunPhase.RUNNING
        assert catalog_calls == []
        assert browser_owner.active_request is None
        assert browser_owner.queued_request is None

        third = deltas[-1].appended
        executor.events.append(StandardRunEvent(
            identity,
            StandardEventKind.FINISHED,
            completed=3,
            total=3,
            artifact=third.artifact,
            detail="Complete · 3 Frames",
            cleanup_status=CleanupStatus.CLEANED,
        ))
        page._drain_executor()

        assert lifecycle.phase is RunPhase.IDLE
        assert not page._batch_terminal.active
        assert page._batch_terminal.latest_frame is None
        assert page._retain_outgoing_display is False
        terminal_owner = page._batch_terminal.presentation
        assert terminal_owner is not None
        assert terminal_owner.run_identity is identity
        assert terminal_owner.frame is third
        assert terminal_owner.painted
        assert not terminal_owner.awaiting_full_raw
        navigation = controller.navigation
        assert navigation.current is third
        assert navigation.selected == (third,)
        assert controller.selection is not None
        assert controller.selection.kind is ContextKind.ACQUISITION
        assert page._processed_browser.terminal_handoff is None
        assert not controller.browse_pending
        assert calls["project"] == 1
        assert calls["commit"] == 1
        assert calls["reconcile"] == 1
        assert calls["traces"] == 1
        assert calls["raw"] == 1
        assert calls["cake"] == 1
        assert calls["sync_detector"] == 0
        assert calls["clear_full"] == 0
        assert calls["request_full"] == 0
        if detector_mode == "full":
            assert page._detector_scope_owner is selection.owner
            assert page._detector_demand_frame is third
        # Batch terminal intentionally projects one exact final frame.  A
        # Waterfall preference therefore remains a one-trace curve rather
        # than activating the multi-row waterfall renderer.
        assert calls["waterfall"] == 0
        assert not scientific.bottom_waterfall_active
        assert scientific.navigation_current_key is third
        assert scientific.trace_history_keys == (third,)
        assert scientific.trace_row_count == 1

        # Batch terminal issues one first-seen refresh followed by the exact
        # terminal catalog request.  Settle both real futures; the final
        # catalog-only refresh may update Browser/status and clear the
        # transient row, but must not project science or demand pixels again.
        terminal_counts = calls.copy()
        _wait_until(qapp, lambda: len(catalog_calls) == 1)
        first_catalog = browser_owner.active_request
        final_catalog = browser_owner.queued_request
        first_wake = browser_owner.active_wake
        assert first_catalog is not None
        assert final_catalog is not None
        assert first_wake is not None
        assert final_catalog is not first_catalog
        catalog_gates[0].set()
        _wait_until(qapp, lambda: first_wake in catalog_wakes)
        page._on_browser_catalog(first_wake)
        _wait_until(qapp, lambda: len(catalog_calls) == 2)
        assert browser_owner.active_request is final_catalog
        assert browser_owner.queued_request is None
        final_wake = browser_owner.active_wake
        assert final_wake is not None and final_wake is not first_wake
        catalog_gates[1].set()
        _wait_until(qapp, lambda: final_wake in catalog_wakes)
        page._on_browser_catalog(final_wake)

        assert browser_owner.active_request is None
        assert browser_owner.queued_request is None
        assert catalog_calls == ["/out", "/out"]
        assert page._processed_browser.transient_frame is None
        assert browser.frame_model.rowCount() == 0
        assert shell.run_controls.readinessLabel.full_text() == (
            "Complete · 3 Frames"
        )
        assert calls == terminal_counts

        # The painted receipt remains available only to suppress an exact
        # duplicate terminal DISPLAY_READY.  It must not hold later ordinary
        # scientific preference edits behind the Batch projection fence.
        before_edit = calls.copy()
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SET_PLOT_OPTION,
            not page._preferences.plot_options.show_legend,
            ("other", "legend"),
        ))
        assert page._batch_terminal.presentation is terminal_owner
        assert not page._batch_terminal.active
        assert page._batch_terminal.latest_frame is None
        assert calls["project"] == before_edit["project"] + 1
        assert calls["commit"] == before_edit["commit"] + 1
        assert calls["reconcile"] == before_edit["reconcile"] + 1
        assert calls["traces"] == before_edit["traces"] + 1

        # A painted-success receipt is not a durable failure fence: ordinary
        # post-success mode interaction retires it and projects normally.
        before_mode = calls.copy()
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SET_PLOT_MODE,
            "Single" if plot_mode != "Single" else "Overlay",
        ))
        assert page._batch_terminal.presentation is None
        assert not page._batch_terminal.active
        assert calls["project"] == before_mode["project"] + 1
        assert calls["commit"] == before_mode["commit"] + 1
        assert calls["reconcile"] == before_mode["reconcile"] + 1
        assert calls["traces"] == before_mode["traces"] + 1
    finally:
        _dispose(page, qapp)


@pytest.mark.parametrize(
    ("kind", "cleanup", "expected_phase", "detail"),
    (
        (
            StandardEventKind.FAILED,
            CleanupStatus.CLEANED,
            RunPhase.FAILED,
            "Batch failed",
        ),
        (
            StandardEventKind.STOPPED,
            CleanupStatus.CLEANED,
            RunPhase.IDLE,
            "Stopped · 3 Frames",
        ),
        (
            StandardEventKind.FINISHED,
            CleanupStatus.CLEANUP_PENDING,
            RunPhase.FAILED,
            "Batch cleanup pending",
        ),
    ),
)
@pytest.mark.parametrize(
    "post_action", ("refresh", "plot_mode", "auto_last", "show_all", "frame"),
)
def test_batch_non_success_terminal_retains_prior_display(
    qapp: QtWidgets.QApplication,
    monkeypatch,
    kind: StandardEventKind,
    cleanup: CleanupStatus,
    expected_phase: RunPhase,
    detail: str,
    post_action: str,
) -> None:
    executor = _Executor()
    page, lifecycle, identity = _active_page(executor)
    try:
        deltas = _batch_display(page, executor, identity)
        controller = page._context_controller
        shell = _shell(page)
        browser = shell.browser
        scientific = shell.scientific
        visible_frames = browser.frame_model.frames
        visible_current = browser._committed_current
        visible_transient = page._processed_browser.transient_frame
        old_projection = page._last_scientific_projection
        old_raw = scientific.raw.image.image
        old_cake = scientific.cake.image.image
        old_items = tuple(scientific.curve.listDataItems())
        calls = {"project": 0, "reconcile": 0}
        real_project = controller.project_navigation
        real_reconcile = scientific.reconcile

        def project(**options):
            calls["project"] += 1
            return real_project(**options)

        def reconcile(*args, **options):
            calls["reconcile"] += 1
            return real_reconcile(*args, **options)

        monkeypatch.setattr(controller, "project_navigation", project)
        monkeypatch.setattr(scientific, "reconcile", reconcile)
        monkeypatch.setattr(page, "_request_browser_catalog", lambda: None)
        page._batch_terminal.begin_run(
            identity, batch_mode=True, visible_progress=page._progress,
        )
        page._retain_outgoing_display = True
        for completed, delta in enumerate(deltas, start=1):
            executor.events.append(StandardRunEvent(
                identity,
                StandardEventKind.FRAME_READY,
                completed=completed,
                total=3,
                artifact=delta.appended.artifact,
                detail=f"Batch {completed}/3",
                frame_key=delta.appended,
                navigation_delta=delta,
            ))
        page._drain_executor()

        terminal_frame = deltas[-1].appended
        executor.events.append(StandardRunEvent(
            identity,
            kind,
            completed=3,
            total=3,
            artifact=terminal_frame.artifact,
            detail=detail,
            cleanup_status=cleanup,
        ))
        page._drain_executor()

        assert lifecycle.phase is expected_phase
        assert page._batch_terminal.active
        assert page._batch_terminal.latest_frame is None
        assert page._batch_terminal.project_progress(page._progress) is page._progress
        owner = page._batch_terminal.presentation
        assert owner is not None
        assert owner.run_identity is identity
        assert owner.frame is None
        assert page._retain_outgoing_display
        assert calls == {"project": 0, "reconcile": 0}
        assert browser.frame_model.frames == visible_frames
        assert browser._committed_current is visible_current
        assert page._processed_browser.transient_frame is visible_transient
        assert page._last_scientific_projection is old_projection
        assert scientific.raw.image.image is old_raw
        assert scientific.cake.image.image is old_cake
        assert tuple(scientific.curve.listDataItems()) == old_items
        terminal_status = (
            "Standard cleanup remains pending"
            if cleanup is CleanupStatus.CLEANUP_PENDING
            else detail
        )
        assert shell.run_controls.readinessLabel.full_text() == terminal_status

        if post_action == "refresh":
            page._refresh_shell()
        elif post_action == "plot_mode":
            page._handle_shell_command(ShellCommand(
                ShellCommandKind.SET_PLOT_MODE, "Overlay",
            ))
        elif post_action == "auto_last":
            page._handle_shell_command(ShellCommand(
                ShellCommandKind.SET_AUTO_LAST, False,
            ))
        elif post_action == "show_all":
            page._handle_shell_command(ShellCommand(
                ShellCommandKind.SHOW_ALL,
            ))
        else:
            first = deltas[0].appended
            page._handle_shell_command(ShellCommand(
                ShellCommandKind.SELECT_FRAME,
                frame=first,
                frames=(first,),
            ))

        assert page._batch_terminal.presentation is owner
        assert page._batch_terminal.active
        assert page._batch_terminal.latest_frame is None
        assert page._batch_terminal.project_progress(page._progress) is page._progress
        assert calls == {"project": 0, "reconcile": 0}
        assert browser.frame_model.frames == visible_frames
        assert browser._committed_current is visible_current
        assert page._processed_browser.transient_frame is visible_transient
        assert page._last_scientific_projection is old_projection
        assert scientific.raw.image.image is old_raw
        assert scientific.cake.image.image is old_cake
        assert tuple(scientific.curve.listDataItems()) == old_items
        assert shell.run_controls.readinessLabel.full_text() == terminal_status
    finally:
        _dispose(page, qapp)


@pytest.mark.parametrize("plot_mode", ("Overlay", "Waterfall"))
def test_batch_full_raw_waits_for_exact_terminal_display_and_paints_once(
    qapp: QtWidgets.QApplication,
    monkeypatch,
    plot_mode: str,
) -> None:
    executor = _Executor()
    page, lifecycle, identity = _active_page(executor)
    try:
        deltas = _batch_display(page, executor, identity)
        controller = page._context_controller
        shell = _shell(page)
        scientific = shell.scientific
        old_projection = page._last_scientific_projection
        old_raw = scientific.raw.image.image
        old_cake = scientific.cake.image.image
        old_items = tuple(scientific.curve.listDataItems())
        assert old_projection is not None
        assert old_raw is not None and old_cake is not None
        assert len(old_items) == 1

        calls = {
            "project": 0,
            "qualify": 0,
            "request": 0,
            "reconcile": 0,
            "waterfall": 0,
        }
        real_project = controller.project_navigation
        real_qualify = controller.qualify_display_event
        real_reconcile = scientific.reconcile
        real_waterfall = scientific.waterfall.render

        def project(**options):
            calls["project"] += 1
            return real_project(**options)

        def qualify(event):
            calls["qualify"] += 1
            return real_qualify(event)

        def request_full():
            calls["request"] += 1
            return object()

        def reconcile(*args, **options):
            calls["reconcile"] += 1
            return real_reconcile(*args, **options)

        def render_waterfall(*args, **options):
            calls["waterfall"] += 1
            return real_waterfall(*args, **options)

        monkeypatch.setattr(controller, "project_navigation", project)
        monkeypatch.setattr(controller, "qualify_display_event", qualify)
        monkeypatch.setattr(controller, "request_full_current", request_full)
        monkeypatch.setattr(
            controller, "full_raw_status", lambda: (False, False, None),
        )
        monkeypatch.setattr(scientific, "reconcile", reconcile)
        monkeypatch.setattr(scientific.waterfall, "render", render_waterfall)
        monkeypatch.setattr(page, "_request_browser_catalog", lambda: None)

        page._preferences = replace(
            page._preferences,
            plot_mode=plot_mode,
            detector_mode="full",
        )
        page._batch_terminal.begin_run(
            identity, batch_mode=True, visible_progress=page._progress,
        )
        page._retain_outgoing_display = True
        for completed, delta in enumerate(deltas, start=1):
            executor.events.append(StandardRunEvent(
                identity,
                StandardEventKind.FRAME_READY,
                completed=completed,
                total=len(deltas),
                artifact=delta.appended.artifact,
                frame_key=delta.appended,
                navigation_delta=delta,
            ))
        page._drain_executor()
        third = deltas[-1].appended

        # Active hydration wakeups are inert before a terminal latch exists.
        selection = controller.selection
        assert selection is not None
        exact_ready = StandardRunEvent(
            identity,
            StandardEventKind.DISPLAY_READY,
            artifact=third.artifact,
            frame_key=third,
            selection_generation=selection.display_generation,
        )
        executor.events.append(exact_ready)
        page._drain_executor()
        assert calls["qualify"] == 0

        executor.events.append(StandardRunEvent(
            identity,
            StandardEventKind.FINISHED,
            completed=3,
            total=3,
            artifact=third.artifact,
            detail="Complete · 3 Frames",
            cleanup_status=CleanupStatus.CLEANED,
        ))
        page._drain_executor()

        owner = page._batch_terminal.presentation
        assert owner is not None
        assert owner.frame is third
        assert owner.awaiting_full_raw
        assert not owner.painted
        assert page._batch_terminal.active
        assert page._batch_terminal.latest_frame is third
        assert calls == {
            "project": 0,
            "qualify": 0,
            "request": 1,
            "reconcile": 0,
            "waterfall": 0,
        }
        assert page._last_scientific_projection is old_projection
        assert scientific.raw.image.image is old_raw
        assert scientific.cake.image.image is old_cake
        assert tuple(scientific.curve.listDataItems()) == old_items

        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SET_CORES, 3,
        ))
        assert page._batch_terminal.presentation is owner
        assert page._batch_terminal.active
        assert calls == {
            "project": 0,
            "qualify": 0,
            "request": 1,
            "reconcile": 0,
            "waterfall": 0,
        }

        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SET_PLOT_MODE,
            "Single" if plot_mode != "Single" else "Overlay",
        ))
        assert page._batch_terminal.presentation is owner
        assert page._batch_terminal.active
        assert calls == {
            "project": 0,
            "qualify": 0,
            "request": 1,
            "reconcile": 0,
            "waterfall": 0,
        }

        foreign_identity = RunIdentity(
            identity.generation + 1, f"{identity.fingerprint}-foreign",
        )
        wrong_frame = deltas[0].appended
        executor.events.extend((
            StandardRunEvent(
                foreign_identity,
                StandardEventKind.DISPLAY_READY,
                artifact=third.artifact,
                frame_key=third,
                selection_generation=selection.display_generation,
            ),
            StandardRunEvent(
                identity,
                StandardEventKind.DISPLAY_READY,
                artifact=wrong_frame.artifact,
                frame_key=wrong_frame,
                selection_generation=selection.display_generation,
            ),
        ))
        page._drain_executor()
        assert calls["qualify"] == 0
        assert page._batch_terminal.presentation is owner

        executor.events.append(replace(
            exact_ready,
            selection_generation=selection.display_generation + 1,
        ))
        page._drain_executor()
        assert calls["qualify"] == 1
        assert page._batch_terminal.presentation is owner

        executor.events.append(exact_ready)
        page._drain_executor()
        painted = page._batch_terminal.presentation
        assert painted is not None
        assert painted.frame is third
        assert painted.painted
        assert not painted.awaiting_full_raw
        assert not page._batch_terminal.active
        assert page._batch_terminal.latest_frame is None
        assert calls == {
            "project": 1,
            "qualify": 2,
            "request": 1,
            "reconcile": 1,
            "waterfall": 0,
        }
        assert lifecycle.phase is RunPhase.IDLE
        assert controller.navigation.current is third
        assert controller.navigation.selected == (third,)
        assert scientific.trace_history_keys == (third,)
        assert scientific.trace_row_count == 1
        assert not scientific.bottom_waterfall_active

        executor.events.extend((exact_ready, exact_ready))
        page._drain_executor()
        assert page._batch_terminal.presentation is painted
        assert calls == {
            "project": 1,
            "qualify": 2,
            "request": 1,
            "reconcile": 1,
            "waterfall": 0,
        }
    finally:
        _dispose(page, qapp)


def test_batch_terminal_accepts_absolute_latest_after_prefix_exceeds_capacity(
    qapp: QtWidgets.QApplication,
    monkeypatch,
) -> None:
    from tests.xdart.scattering.test_e3_context_contract import _view

    executor = _Executor()
    page, lifecycle, identity = _active_page(executor)
    try:
        deltas = _batch_display(page, executor, identity)
        acquisition = executor.acquisition_context(identity)
        assert acquisition is not None
        display = acquisition.publication_store
        retired = display.catalog.resize(2)
        assert retired == (deltas[0].appended,)
        page._context_controller.adopt_acquisition(identity)
        assert tuple(
            frame.work_ordinal
            for frame in page._context_controller.navigation.frames
        ) == (2, 3)

        view = _view(4, 4.0)
        record = FrameRecord.from_view(view)
        live = display.append_navigation("run.a", "/out/a.nxs", 4)
        owner = display.artifacts[live.appended.artifact]
        publication = FramePublication(
            view,
            record=record,
            source_identity=f"{view.source_path}#4",
            scan_key=live.appended.source_scan,
        )
        display.retain_frame(
            owner,
            live.appended,
            record,
            publication,
            source_identity=publication.source_identity,
            frame_mask_qualified=False,
        )
        display.put_payload(StandardDisplayPayload(
            0, live.appended, "Standard · run.a · frame 4", view,
        ))
        assert live.appended.work_ordinal == 4

        controller = page._context_controller
        projects = []
        real_project = controller.project_navigation

        def project(**options):
            projects.append(options)
            return real_project(**options)

        monkeypatch.setattr(controller, "project_navigation", project)
        monkeypatch.setattr(page, "_request_browser_catalog", lambda: None)
        page._preferences = replace(
            page._preferences,
            plot_mode="Overlay",
            detector_mode="thumbnail",
        )
        page._batch_terminal.begin_run(
            identity, batch_mode=True, visible_progress=page._progress,
        )
        page._retain_outgoing_display = True
        executor.events.append(StandardRunEvent(
            identity,
            StandardEventKind.FRAME_READY,
            completed=4,
            total=4,
            artifact=live.appended.artifact,
            frame_key=live.appended,
            navigation_delta=live,
        ))
        page._drain_executor()
        assert projects == []
        assert page._batch_terminal.latest_frame is live.appended

        executor.events.append(StandardRunEvent(
            identity,
            StandardEventKind.FINISHED,
            completed=4,
            total=4,
            artifact=live.appended.artifact,
            detail="Complete · 4 Frames",
            cleanup_status=CleanupStatus.CLEANED,
        ))
        page._drain_executor()

        terminal = page._batch_terminal.presentation
        assert terminal is not None
        assert terminal.frame is live.appended
        assert terminal.painted
        assert not page._batch_terminal.active
        assert page._batch_terminal.latest_frame is None
        assert len(projects) == 1
        assert lifecycle.phase is RunPhase.IDLE
        assert controller.navigation.current is live.appended
        assert controller.navigation.selected == (live.appended,)
        assert _shell(page).scientific.trace_history_keys == (live.appended,)
    finally:
        _dispose(page, qapp)


@pytest.mark.parametrize(
    "terminal_case", ("zero", "owned_historical", "request_refused"),
)
def test_batch_terminal_without_exact_latest_preserves_prior_science(
    qapp: QtWidgets.QApplication,
    monkeypatch,
    terminal_case: str,
) -> None:
    executor = _Executor()
    page, lifecycle, identity = _active_page(executor)
    try:
        deltas = _batch_display(page, executor, identity)
        controller = page._context_controller
        shell = _shell(page)
        scientific = shell.scientific
        browser = shell.browser
        old_projection = page._last_scientific_projection
        old_raw = scientific.raw.image.image
        old_cake = scientific.cake.image.image
        old_items = tuple(scientific.curve.listDataItems())
        assert old_projection is not None
        assert old_raw is not None and old_cake is not None
        assert len(old_items) == 1
        old_item = old_items[0]
        old_x, old_y = old_item.xData, old_item.yData
        if terminal_case == "zero":
            controller._runtime._acquisition_navigation = (
                FrameNavigationProjection()
            )

        calls = {
            "project": 0,
            "reconcile": 0,
            "clear_full": 0,
            "request_full": 0,
        }
        real_project = controller.project_navigation
        real_reconcile = scientific.reconcile

        def project(**options):
            calls["project"] += 1
            return real_project(**options)

        def reconcile(*args, **options):
            calls["reconcile"] += 1
            return real_reconcile(*args, **options)

        def clear_full():
            calls["clear_full"] += 1
            return True

        def request_full():
            calls["request_full"] += 1
            return None

        monkeypatch.setattr(controller, "project_navigation", project)
        monkeypatch.setattr(scientific, "reconcile", reconcile)
        monkeypatch.setattr(controller, "clear_full_raw", clear_full)
        monkeypatch.setattr(
            controller, "request_full_current", request_full,
        )
        if terminal_case == "request_refused":
            monkeypatch.setattr(
                controller,
                "full_raw_status",
                lambda: (False, False, None),
            )
        page._preferences = replace(
            page._preferences, detector_mode="full",
        )
        detector_preferences = page._preferences
        selection = controller.selection
        assert selection is not None
        detector_owner = selection.owner
        detector_frame = deltas[0].appended
        page._detector_scope_owner = detector_owner
        page._detector_demand_frame = detector_frame
        page._processed_browser.set_directory(
            os.path.dirname(deltas[0].appended.artifact),
            explicit=False,
        )
        page._batch_terminal.begin_run(
            identity, batch_mode=True, visible_progress=page._progress,
        )
        page._retain_outgoing_display = True
        visible_frames = browser.frame_model.frames
        visible_current = browser._committed_current
        visible_status = shell.run_controls.readinessLabel.full_text()
        visible_transient = page._processed_browser.transient_frame
        if terminal_case in {"owned_historical", "request_refused"}:
            for completed, delta in enumerate(deltas, start=1):
                frame = delta.appended
                executor.events.append(StandardRunEvent(
                    identity,
                    StandardEventKind.FRAME_READY,
                    completed=completed,
                    total=3,
                    artifact=frame.artifact,
                    detail=f"Batch {completed}/3",
                    frame_key=frame,
                    navigation_delta=delta,
                ))
                page._drain_executor()
                assert page._progress.completed == completed
                assert page._progress.total == 3
                assert browser.frame_model.frames == visible_frames
                assert browser._committed_current is visible_current
                assert shell.run_controls.readinessLabel.full_text() == (
                    visible_status
                )
                assert page._processed_browser.transient_frame is visible_transient
            if terminal_case == "owned_historical":
                # This is an exact owned member, but not the terminal event's
                # exact latest frame.  Membership alone must never qualify it.
                earlier = deltas[0].appended
                page._batch_terminal.record_frame(
                    StandardRunEvent(
                        identity,
                        StandardEventKind.FRAME_READY,
                        frame_key=earlier,
                    ),
                    earlier,
                )
            completed = total = 3
            artifact = deltas[-1].appended.artifact
        else:
            completed = total = 0
            artifact = "/out/a.nxs"
        executor.events.append(StandardRunEvent(
            identity,
            StandardEventKind.FINISHED,
            completed=completed,
            total=total,
            artifact=artifact,
            detail=f"Complete · {completed} Frames",
            cleanup_status=CleanupStatus.CLEANED,
        ))
        page._drain_executor()

        assert lifecycle.phase is RunPhase.IDLE
        assert page._batch_terminal.active
        assert page._batch_terminal.latest_frame is None
        terminal_owner = page._batch_terminal.presentation
        assert terminal_owner is not None
        assert terminal_owner.run_identity is identity
        assert terminal_owner.frame is None
        assert page._retain_outgoing_display is True
        assert calls == {
            "project": 0,
            "reconcile": 0,
            "clear_full": 0,
            "request_full": (
                1 if terminal_case == "request_refused" else 0
            ),
        }
        assert browser.frame_model.frames == visible_frames
        assert browser._committed_current is visible_current
        assert page._processed_browser.transient_frame is visible_transient
        assert shell.run_controls.readinessLabel.full_text() == (
            f"Complete · {completed} Frames"
        )
        if terminal_case == "request_refused":
            assert not page._preferences.detector_pending
            assert "refused" in page._preferences.detector_diagnostic.lower()
            assert "refused" in page._notice_text.lower()
        else:
            assert page._preferences is detector_preferences
        assert page._detector_scope_owner is detector_owner
        assert page._detector_demand_frame is (
            deltas[-1].appended
            if terminal_case == "request_refused"
            else detector_frame
        )
        assert page._last_scientific_projection is old_projection
        assert scientific.raw.image.image is old_raw
        assert scientific.cake.image.image is old_cake
        assert tuple(scientific.curve.listDataItems()) == (old_item,)
        assert old_item.xData is old_x and old_item.yData is old_y
    finally:
        _dispose(page, qapp)


def test_run_retries_transient_prior_display_cleanup_without_second_click(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch,
) -> None:
    executor = _Executor()
    source = image_series_spec(tmp_path / "frame_0001.tif")
    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                source_spec=source,
                poni_file=str(tmp_path / "calibration.poni"),
                save_path=str(tmp_path / "output.nxs"),
                output_mode="Overwrite",
            )
        ),
        lifecycle=ScatteringCoordinator(),
        sources=_Sources(),
        executor=executor,
    )
    attempts: list[object] = []

    def fail_once(receipt):
        attempts.append(receipt)
        return len(attempts) > 1

    monkeypatch.setattr(
        page._context_controller,
        "apply_display_retirement",
        fail_once,
    )
    try:
        _shell(page).commandRequested.emit(
            ShellCommand(ShellCommandKind.RUN_ACTION)
        )
        page._drain_executor()
        assert executor.start_calls == 0
        assert page._admission is not None
        assert "Prior display cleanup" not in page._notice_text
        assert (
            _shell(page).run_controls.readinessLabel.text()
            == "Finishing prior display cleanup…"
        )

        page._drain_executor()
        assert executor.start_calls == 1
        assert page._admission is None
        assert attempts[0] is attempts[1]
    finally:
        _dispose(page, qapp)


def test_clean_output_release_projects_pending_display_retirement(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch,
) -> None:
    executor = _Executor()
    lifecycle = ScatteringCoordinator()
    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                source_spec=image_series_spec(tmp_path / "frame_0001.tif"),
                poni_file=str(tmp_path / "calibration.poni"),
                save_path=str(tmp_path / "output.nxs"),
                output_mode="Overwrite",
            )
        ),
        lifecycle=lifecycle,
        sources=_Sources(),
        executor=executor,
    )
    allow_retirement = False

    def apply_retirement(_receipt) -> bool:
        return allow_retirement

    monkeypatch.setattr(
        page._context_controller,
        "apply_display_retirement",
        apply_retirement,
    )
    try:
        shell = _shell(page)
        shell.commandRequested.emit(
            ShellCommand(ShellCommandKind.RUN_ACTION)
        )
        page._drain_executor()
        assert page._admission is not None

        shell.commandRequested.emit(ShellCommand(ShellCommandKind.STOP))

        state = page._admission_state
        assert state is not None
        assert state.releasing
        assert state.release_receipt is not None
        assert state.release_receipt.cleanup_status is CleanupStatus.CLEANED
        assert lifecycle.phase is RunPhase.IDLE
        assert (
            shell.run_controls.readinessLabel.text()
            == "Finishing prior display cleanup…"
        )

        allow_retirement = True
        page._drain_executor()
        assert page._admission is None
    finally:
        _dispose(page, qapp)


def test_native_1d_axis_follows_only_the_exact_run_first_paint(
    qapp: QtWidgets.QApplication,
) -> None:
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=image_series_spec(Path("frame_0001.tif")),
            poni_file="calibration.poni",
            save_path="output.nxs",
            output_mode="Overwrite",
            processing_mode="Int 2D",
            bai_1d_args={"unit": "qip_A^-1"},
            gi=GIIntent(
                enabled=True,
                incidence_motor="Manual",
                mode_1d="q_ip",
                mode_2d="qip_qoop",
            ),
        )),
        lifecycle=ScatteringCoordinator(),
        sources=_Sources(),
        executor=_Executor(),
    )
    try:
        page._preferences = ScientificPreferences(
            plot_axis="q_ip",
            share_axis=False,
        )
        outgoing = page._preferences

        page._on_field_value(INT_1D_AXIS, "Q")
        configuration = page._intents.snapshot().thaw().freeze()
        target = RunIdentity(41, "target-q")
        wrong = RunIdentity(42, "wrong-q")

        assert page._preferences.plot_axis == outgoing.plot_axis
        assert page._preferences.share_axis == outgoing.share_axis
        assert page._preferences.slice_pins == outgoing.slice_pins
        assert page._native_plot_axis_transition is not None
        assert (
            page._native_plot_axis_transition.origin_axis,
            page._native_plot_axis_transition.target_axis,
            page._native_plot_axis_transition.run_identity,
        ) == ("q_ip", "Q", None)
        assert configuration.gi.mode_1d == "q_total"
        assert configuration.bai_1d_args["unit"] == "q_A^-1"
        page._bind_native_plot_axis_to_run(target, configuration)
        assert not page._consume_native_plot_axis_transition(wrong)
        assert page._preferences.plot_axis == "q_ip"
        page._release_native_plot_axis_transition_for_retry(target)
        assert page._native_plot_axis_transition is not None
        assert page._native_plot_axis_transition.run_identity is None
        retry = RunIdentity(45, "retry-q")
        page._bind_native_plot_axis_to_run(retry, configuration)
        assert page._consume_native_plot_axis_transition(retry)
        assert page._preferences.plot_axis == "Q"
        assert page._native_plot_axis_transition is None

        # A manual preference change after launch is authoritative.
        page._on_field_value(INT_1D_AXIS, "Qip")
        configuration = page._intents.snapshot().thaw().freeze()
        manual_target = RunIdentity(43, "target-qip")
        page._bind_native_plot_axis_to_run(manual_target, configuration)
        page._preferences = replace(page._preferences, plot_axis="2theta")
        assert not page._consume_native_plot_axis_transition(manual_target)
        assert page._preferences.plot_axis == "2theta"

        # Explicit Share Axis is equally authoritative at first paint.
        page._preferences = replace(
            page._preferences,
            plot_axis="q_ip",
            share_axis=False,
        )
        page._on_field_value(INT_1D_AXIS, "Q")
        configuration = page._intents.snapshot().thaw().freeze()
        shared_target = RunIdentity(44, "target-shared-q")
        page._bind_native_plot_axis_to_run(shared_target, configuration)
        page._preferences = replace(page._preferences, share_axis=True)
        assert not page._consume_native_plot_axis_transition(shared_target)
        assert page._preferences.plot_axis == "q_ip"
        assert page._preferences.share_axis

        # Normal DISPLAY_READY and the Batch terminal paint use the same
        # exact-identity transition seam.
        import inspect
        assert "_consume_native_plot_axis_transition" in inspect.getsource(
            ScatteringWorkspace._drain_executor
        )
        assert "_consume_native_plot_axis_transition" in inspect.getsource(
            ScatteringWorkspace._paint_batch_terminal
        )
        assert "_release_native_plot_axis_transition_for_retry" in (
            inspect.getsource(ScatteringWorkspace._stop_run)
        )
        assert "_release_native_plot_axis_transition_for_retry" in (
            inspect.getsource(ScatteringWorkspace._render_start_outcome)
        )
    finally:
        _dispose(page, qapp)
