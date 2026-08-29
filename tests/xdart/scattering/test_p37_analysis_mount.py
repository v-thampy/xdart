from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import importlib
import os
from pathlib import Path
import struct
import sys
from threading import Event
import time
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.adapters.external_operation import OperationSlot
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.operation_values import (
    OperationContextStamp,
    OperationIdentity,
    OperationProgress,
    OperationTerminal,
    OperationTerminalStatus,
    OperationUpdate,
)
from xdart.gui.tabs.scattering.shell_values import (
    AxisProjection,
    ShellCommand,
    ShellCommandKind,
    ScientificProjection,
    TraceProjection,
)
from xdart.gui.tabs.scattering.tools_view import ToolsView
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.events import CleanupStatus, RunIdentity
from xdart.gui.tabs.scattering.metadata_operations import MetadataLifecycle
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest
from xdart.gui.tabs.scattering.processed_browser import (
    ReintegrateReloadDirective,
)
from xdart.gui.tabs.scattering.workspace_operations import (
    WorkspaceRefreshEffect,
)
from xrd_tools.analysis.plans import RoiSignal
from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition, AnalysisSourceReceipt, CandidateProjection,
    MetadataColumn, MetadataTablePlan, MetadataTableResult,
    MetadataTableRequalificationPlan, MetadataTableRequalificationResult,
    run_metadata_table_requalification,
    RoiPreviewPlan, RoiPreviewResult, RoiScanPlan, RoiScanResult,
    ScanPlotPlan, ScanPlotResult,
)
from xrd_tools.analysis.display_fit_operations import (
    CifAssetReceipt, DisplayedPeakFitPlan, DisplayedPeakFitResult,
    DisplayedPhaseFitPlan, DisplayedPhaseFitResult,
)
from xrd_tools.core.roi import RoiSpec
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent


def _set_pending_reintegrate_reload(
    page, target: str = "/tmp/current.nexus",
) -> ReintegrateReloadDirective:
    request = BrowseLoadRequest("pending-reintegrate", 1, target)
    directive = ReintegrateReloadDirective(request, target)
    assert page._processed_browser.adopt_reload(directive) is directive
    return directive


def _clear_pending_reintegrate_reload(page) -> None:
    directive = page._processed_browser.pending_reintegrate_reload
    if directive is not None:
        assert page._processed_browser.retire_reload(directive)


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _frame(label: int = 3) -> DisplayFrameKey:
    return DisplayFrameKey(RunIdentity(2, "run-fingerprint"), "scan-7",
                           "/tmp/a.nxs", label, 9)


def _projection(*, epoch: float | None = 1700000000.25):
    current, other = _frame(), _frame(4)
    q = np.array([1.0, 2.0, 3.0], dtype="<f8")
    return current, ScientificProjection(
        traces=(
            TraceProjection(other, AxisProjection(q, "q", "A^-1"), q + 4, "other"),
            TraceProjection(current, AxisProjection(q, "q", "A^-1"), q + 7,
                            "current", epoch),
        )
    )


def _finish(slot: OperationSlot, identity: OperationIdentity) -> OperationUpdate:
    worker = slot._worker
    assert worker is not None
    worker.join(2)
    assert not worker.is_alive()
    update = slot.poll(identity)
    assert type(update) is OperationUpdate and update.terminal is not None
    return update


def _source(path: str = "/tmp/input.nxs"):
    return SimpleNamespace(
        source=path, selection="exact", kind=None, recursive=False, entry=None,
        scan=None, image_dir=None, image_stem=None, source_root=None,
        metadata_format=None,
    )


def _file_revision(path):
    state = Path(path).stat()
    return (
        state.st_mode, state.st_dev, state.st_ino, state.st_size,
        state.st_mtime_ns,
    )


def _receipt(path="/tmp/input.nxs", *, fingerprint="source-fp", labels=(1, 2, 3),
             primary_post_state=None):
    spec = path if type(path) is SourceSpec else SourceSpec(
        path, SourceKind.PROCESSED_NEXUS)
    state = (0, 0, 0, 1, 1)
    post_state = state if primary_post_state is None else primary_post_state
    return AnalysisSourceReceipt(
        "analysis-source-v2", spec, "spec-digest", Path(str(spec.uri)),
        Path(str(spec.uri)), spec.kind, spec.entry, None, state, post_state,
        tuple(labels), "labels-digest", "catalog-digest", ("persisted",),
        (), fingerprint,
    )


def _table(path="/tmp/input.nxs", *, fingerprint="table-fp",
           source_fingerprint="source-fp", primary_post_state=None):
    receipt = _receipt(
        path, fingerprint=source_fingerprint,
        primary_post_state=primary_post_state,
    )
    frame = np.array([1.0, 2.0, 3.0])
    motor = np.array([10.0, 11.0, 12.0])
    signal = np.array([4.0, 5.0, 6.0])
    columns = tuple(MetadataColumn(name, "numeric", "nan", values, None,
                                   values.nbytes) for name, values in (
        ("frame_index", frame), ("motor", motor), ("signal", signal)))
    return MetadataTableResult(
        AnalysisDisposition.COMPLETED, "OK", receipt=receipt,
        labels=receipt.labels, columns=columns,
        selected_scanned_positioner="motor", table_fingerprint=fingerprint,
        storage_bytes=sum(column.storage_bytes for column in columns),
    )


def _trace():
    from xdart.gui.tabs.scattering.analysis_mount import displayed_trace_input
    current, projection = _projection()
    return displayed_trace_input(projection, current)


def _page():
    return ScatteringWorkspace(
        intents=RunIntentStore(RunIntent()), lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter())


def _close_page(page, qapp):
    page.close_workspace(); page.deleteLater(); qapp.processEvents()


def _returned(identity, payload, *, stale=False):
    return OperationUpdate(identity, terminal=OperationTerminal(
        identity, OperationTerminalStatus.RETURNED, payload=payload), stale=stale)


def _start_metadata_process(
    page,
    identity,
    plan,
    request,
    *,
    target="metadata",
    candidate=None,
):
    if candidate is not None:
        plan = MetadataTableRequalificationPlan(
            candidate.receipt,
            candidate.table_fingerprint,
        )
    token = page._metadata_operations.dialog_identity(target)
    assert token is not None
    captured = page._metadata_operations.capture(
        plan,
        request,
        target=target,
        dialog=token,
        candidate=candidate,
    )
    assert captured is not None
    process = page._metadata_operations.start(
        captured,
        identity,
        page._metadata_context_stamp(),
    )
    assert process is not None
    return process


def _check_item(widget, name, checked=True):
    matches = widget.findItems(name, QtCore.Qt.MatchFlag.MatchExactly)
    assert len(matches) == 1
    matches[0].setCheckState(QtCore.Qt.CheckState.Checked if checked else
                             QtCore.Qt.CheckState.Unchecked)


def test_p37b_literal_commands_open_four_singletons_and_share_scan_roi_mount() -> None:
    # This is the genuine parent RED: the terminal P3-7A parent has only three
    # tools and no ROI route.  The production fix adds the literal fourth row.
    assert ToolsView._TOOLS == (
        ("\u2227 Peak Fitting", "peak_fitting"),
        ("\u2248 Phase Fitting", "phase_fitting"),
        ("\u25a4 Plot Metadata", "plot_metadata"),
        ("\u25a3 ROI Statistics", "roi_statistics"),
    )
    from xdart.gui.tabs.scattering.analysis_mount import MOUNT_TARGETS

    assert MOUNT_TARGETS == {
        (ShellCommandKind.SHOW_METADATA, ""): "metadata",
        (ShellCommandKind.LAUNCH_TOOL, "plot_metadata"): "scan_roi",
        (ShellCommandKind.LAUNCH_TOOL, "roi_statistics"): "scan_roi",
        (ShellCommandKind.LAUNCH_TOOL, "peak_fitting"): "peak",
        (ShellCommandKind.LAUNCH_TOOL, "phase_fitting"): "phase",
        (ShellCommandKind.ANALYSIS_ACTION, "scan_plot"): "scan_roi",
        (ShellCommandKind.ANALYSIS_ACTION, "roi_stats"): "scan_roi",
        (ShellCommandKind.ANALYSIS_ACTION, "peak_fit"): "peak",
        (ShellCommandKind.ANALYSIS_ACTION, "phase_fit"): "phase",
    }


def test_p37b_six_specific_requests_use_the_one_analysis_slot(monkeypatch) -> None:
    import xrd_tools.analysis.scan_operations as scan
    import xrd_tools.analysis.display_fit_operations as fits

    table, trace = _table(), _trace(); receipt = table.receipt
    assert receipt is not None
    roi_signal = RoiSignal(RoiSpec.full_frame("roi"), name="roi")
    plans = (
        MetadataTablePlan(receipt.source_spec),
        ScanPlotPlan("motor", ("signal",), None),
        RoiPreviewPlan.from_table(table, label=1),
        RoiScanPlan.from_table(table, signals=(roi_signal,)),
        DisplayedPeakFitPlan(trace, fit_bounds=(1.0, 3.0)),
        DisplayedPhaseFitPlan(trace, (Path("/tmp/a.cif"),), (None,),
                              ("a",), 1.0),
    )
    results = (
        table,
        ScanPlotResult(AnalysisDisposition.COMPLETED, "OK",
            table_fingerprint=table.table_fingerprint, x_name="motor",
            x=np.arange(3.0), trace_names=("signal",),
            original_identities=("signal",), traces=(np.arange(3.0),)),
        RoiPreviewResult(AnalysisDisposition.COMPLETED, "OK", receipt=receipt,
            table_fingerprint=table.table_fingerprint, labels=table.labels,
            label=1, image=np.ones((2, 2))),
        RoiScanResult(AnalysisDisposition.COMPLETED, "OK", receipt=receipt,
            table_fingerprint=table.table_fingerprint, signal_names=("roi",)),
        DisplayedPeakFitResult(AnalysisDisposition.COMPLETED, "OK",
            trace_fingerprint=trace.trace_fingerprint,
            trace_receipt=trace.receipt),
        DisplayedPhaseFitResult(AnalysisDisposition.COMPLETED, "OK",
            trace_fingerprint=trace.trace_fingerprint,
            trace_receipt=trace.receipt),
    )
    seen = []
    monkeypatch.setattr(scan, "run_metadata_table", lambda plan, **kw:
        seen.append(("metadata", plan, kw["cancel_token"])) or results[0])
    monkeypatch.setattr(scan, "run_scan_plot", lambda plan, table, **kw:
        seen.append(("scan_plot", plan, table, kw["cancel_token"])) or results[1])
    monkeypatch.setattr(scan, "run_roi_preview", lambda plan, **kw:
        seen.append(("roi_preview", plan, kw["cancel_token"])) or results[2])
    monkeypatch.setattr(scan, "run_roi_scan", lambda plan, **kw:
        seen.append(("roi_scan", plan, kw["cancel_token"])) or results[3])
    monkeypatch.setattr(fits, "run_displayed_peak_fit", lambda plan, **kw:
        seen.append(("peak", plan, kw["cancel_token"])) or results[4])
    monkeypatch.setattr(fits, "run_displayed_phase_fit", lambda plan, **kw:
        seen.append(("phase", plan, kw["cancel_token"])) or results[5])
    slot, stamp = OperationSlot(), OperationContextStamp(0)
    launches = (
        lambda: slot.begin_metadata(plans[0], stamp),
        lambda: slot.begin_scan_plot(plans[1], table, None, stamp),
        lambda: slot.begin_roi_preview(plans[2], stamp),
        lambda: slot.begin_roi_scan(plans[3], stamp),
        lambda: slot.begin_peak_fit(plans[4], stamp),
        lambda: slot.begin_phase_fit(plans[5], stamp),
    )
    for index, launch in enumerate(launches):
        identity = launch(); assert type(identity) is OperationIdentity
        update = _finish(slot, identity)
        assert update.terminal.payload is results[index]
    assert tuple(row[0] for row in seen) == (
        "metadata", "scan_plot", "roi_preview", "roi_scan", "peak", "phase")
    assert all(type(row[-1]) is Event for row in seen)
    assert not slot.owned


def test_p37b_only_exact_idle_without_writer_or_cleanup_can_start() -> None:
    from xdart.gui.tabs.scattering.analysis_mount import analysis_start_allowed

    page = SimpleNamespace(
        _closing=False, _closed=False, _admission_state=None,
        _analysis_slot=SimpleNamespace(owned=False),
        _workspace_operations=SimpleNamespace(owned=False),
        _lifecycle=SimpleNamespace(phase=SimpleNamespace(value="idle"),
                                   active_run_identity=None,
                                   attempt_run_identity=None),
        _context_controller=SimpleNamespace(
            browse_pending=False, viewer_1d_cleanup_pending=False,
            viewer_2d_cleanup_pending=False),
    )
    page._experiment_operation_busy = lambda: page._workspace_operations.owned
    assert analysis_start_allowed(page)
    for owner, name, bad in (
        (page, "_closing", True),
        (page, "_admission_state", object()),
        (page._analysis_slot, "owned", True),
        (page._workspace_operations, "owned", True),
        (page._lifecycle, "active_run_identity", object()),
        (page._context_controller, "browse_pending", True),
    ):
        old = getattr(owner, name); setattr(owner, name, bad)
        assert not analysis_start_allowed(page)
        setattr(owner, name, old)


def test_p37b_metadata_direct_and_scheduled_results_match_and_render_detached(
        monkeypatch, qapp) -> None:
    from xdart.gui.tabs.scattering.analysis_mount import analysis_request_facts

    page = _page()
    try:
        page._open_analysis_mount("metadata")
        metadata_dialog = page._metadata_dialog
        plan = MetadataTablePlan("/tmp/current.nxs")
        facts = analysis_request_facts(plan)
        identity = OperationIdentity(41)
        requalification_identity = OperationIdentity(42)
        with monkeypatch.context() as patch:
            patch.setattr(page, "_current_analysis_request",
                          lambda kind, target: facts)
            patch.setattr(page._analysis_slot, "begin_metadata",
                          lambda actual, stamp: identity)
            patch.setattr(
                page._analysis_slot, "begin_metadata_requalification",
                lambda actual, stamp: requalification_identity,
            )
            assert page._submit_metadata(
                plan, target="metadata", request=facts,
            ) is identity
            current = _table("/tmp/current.nxs", fingerprint="current-table")
            assert page._consume_analysis_update(_returned(identity, current))
            assert page._metadata_result is None
            assert (
                page._metadata_operations.active_identity
                is requalification_identity
            )
            validation = MetadataTableRequalificationResult(
                AnalysisDisposition.COMPLETED, "OK", current.receipt,
                current.table_fingerprint,
            )
            assert page._consume_analysis_update(
                _returned(requalification_identity, validation)
            )
        assert page._metadata_result is current
        assert metadata_dialog.table.rowCount() == len(current.labels)
        assert metadata_dialog.status.text() == "OK"

        page._open_analysis_mount("scan_roi")
        dialog = page._scan_roi_dialog
        dialog.source_widget.dir_check.setChecked(True)
        dialog.source_widget.path_edit.setText("/tmp/folder")
        syntax = dialog.source_widget.external_source_syntax()
        assert syntax[0] == "directory"
        launches = []
        identities = iter((OperationIdentity(43), OperationIdentity(44)))
        selected_requalification = OperationIdentity(45)
        monkeypatch.setattr(page._analysis_slot, "begin_metadata",
            lambda actual, stamp: launches.append(actual) or next(identities))
        monkeypatch.setattr(
            page._analysis_slot, "begin_metadata_requalification",
            lambda actual, stamp: selected_requalification,
        )
        page._scan_analysis_action("metadata", syntax)
        candidate_identity = page._metadata_operations.active_identity
        assert launches[-1].selection == "directory"
        spec = SourceSpec("/tmp/folder/scan.nxs", SourceKind.PROCESSED_NEXUS)
        refused = MetadataTableResult(
            AnalysisDisposition.REFUSED, "SOURCE_SELECTION_REQUIRED",
            candidates=(CandidateProjection(spec, "candidate-fp", 32),))
        assert page._consume_analysis_update(_returned(candidate_identity, refused))
        assert dialog._vnext_table_result is None
        assert dialog.source_widget.scan_combo.currentText() == "Choose a scan…"
        assert dialog.source_widget.external_source_syntax() is None
        assert page._metadata_result is current

        dialog.source_widget.scan_combo.setCurrentIndex(1)
        exact_identity = page._metadata_operations.active_identity
        assert launches[-1].selection == "exact" and launches[-1].source == spec
        selected = _table(spec, fingerprint="selected-table")
        assert page._consume_analysis_update(_returned(exact_identity, selected))
        assert dialog._vnext_table_result is None
        assert page._consume_analysis_update(_returned(
            selected_requalification,
            MetadataTableRequalificationResult(
                AnalysisDisposition.COMPLETED, "OK", selected.receipt,
                selected.table_fingerprint,
            ),
        ))
        assert dialog._vnext_table_result is selected
        assert page._metadata_result is selected
        assert metadata_dialog.table.rowCount() == len(current.labels)
    finally:
        _close_page(page, qapp)


def test_metadata_button_requeries_after_current_artifact_changes(
        monkeypatch, qapp) -> None:
    from tests.xdart.scattering.test_e1b2_page_command_boundaries import (
        _Executor,
        _active_page,
        _dispose,
        _paced_frame_events,
    )

    executor = _Executor()
    page, _, identity = _active_page(executor)
    try:
        events = _paced_frame_events(page, executor, identity, 1)
        executor.events.extend(events)
        page._drain_executor()
        current = page._context_controller.navigation.current
        assert current is not None

        stale = _table("/tmp/stale-metadata.nxs")
        launches = []
        monkeypatch.setattr(
            page,
            "_submit_metadata",
            lambda plan, **kwargs: launches.append((plan, kwargs)),
        )
        page._metadata_result = stale

        page._open_analysis_mount("metadata")

        assert len(launches) == 1
        assert str(launches[0][0].source) == current.artifact
        assert page._metadata_result is None
        dialog = page._metadata_dialog
        assert dialog is not None
        assert dialog.table.rowCount() == 0
        assert dialog.table.columnCount() == 0
        assert dialog.status.text() == ""
    finally:
        _dispose(page, qapp)


def test_metadata_first_click_survives_viewer_cleanup_with_same_dialog(
    monkeypatch, qapp,
) -> None:
    from tests.xdart.scattering.test_e3_context_contract import _browse
    from xdart.modules.display_context import ContextKind, new_context_token

    page = _page()
    request, browse = _browse(
        new_context_token(ContextKind.BROWSE), 1, scan_key="loaded-nexus",
    )
    page._context_controller._runtime.adopt_browse(browse, request)
    page._refresh_shell()
    held = [True]
    controller_type = type(page._context_controller)
    monkeypatch.setattr(
        controller_type,
        "viewer_2d_cleanup_pending",
        property(lambda _controller: held[0]),
    )
    launches = []
    identity = OperationIdentity(45)
    monkeypatch.setattr(
        page._analysis_slot,
        "begin_metadata",
        lambda plan, stamp: launches.append((plan, stamp)) or identity,
    )
    try:
        page._shell.browser.metadata.click()
        qapp.processEvents()

        dialog = page._metadata_dialog
        deferred = page._metadata_operations.deferred
        assert dialog is not None and dialog.isVisible()
        assert deferred is not None
        assert deferred.dialog is page._metadata_operations.dialog_identity(
            "metadata"
        )
        assert deferred.request == page._current_analysis_request(
            "metadata", "metadata",
        )
        assert launches == []
        assert page._polling_needed()

        held[0] = False
        page._drain_executor()

        assert page._metadata_dialog is dialog and dialog.isVisible()
        assert len(launches) == 1
        assert page._metadata_operations.active_identity is identity
        assert page._metadata_operations.deferred is None
        page._drain_executor()
        assert len(launches) == 1
    finally:
        _close_page(page, qapp)


def test_metadata_progress_and_requalification_refresh_only_the_same_dialog(
    monkeypatch, qapp,
) -> None:
    from tests.xdart.scattering.test_e3_context_contract import _browse
    from xdart.modules.display_context import ContextKind, new_context_token
    import xrd_tools.analysis.scan_operations as scan_operations

    page = _page()
    request, browse = _browse(
        new_context_token(ContextKind.BROWSE), 1, scan_key="refresh-scope",
    )
    page._context_controller._runtime.adopt_browse(browse, request)
    page._refresh_shell()
    table = _table(request.source_path, fingerprint="refresh-table")
    metadata_entered, metadata_release = Event(), Event()
    requalification_entered, requalification_release = Event(), Event()

    def run_metadata(plan, *, cancel_token, progress_callback):
        assert plan.source == request.source_path
        assert not cancel_token.is_set()
        progress_callback(1, 2)
        metadata_entered.set()
        assert metadata_release.wait(2)
        progress_callback(2, 2)
        return table

    def run_requalification(plan, *, cancel_token, progress_callback):
        assert plan.receipt == table.receipt
        assert plan.table_fingerprint == table.table_fingerprint
        assert not cancel_token.is_set()
        progress_callback(1, 2)
        requalification_entered.set()
        assert requalification_release.wait(2)
        progress_callback(2, 2)
        return MetadataTableRequalificationResult(
            AnalysisDisposition.COMPLETED,
            "OK",
            table.receipt,
            table.table_fingerprint,
        )

    monkeypatch.setattr(
        scan_operations, "run_metadata_table", run_metadata,
    )
    monkeypatch.setattr(
        scan_operations,
        "run_metadata_table_requalification",
        run_requalification,
    )
    refreshes = []
    controls = page._shell.controls
    original_controls_reconcile = controls.reconcile
    original_browser = page._shell.browser.reconcile
    original_heavy_residency = (
        page._shell.browser.reconcile_heavy_residency
    )
    original_scientific = page._shell.scientific.reconcile

    def controls_reconcile(state):
        refreshes.append("controls")
        return original_controls_reconcile(state)

    def browser_refresh(*args, **kwargs):
        refreshes.append("browser")
        return original_browser(*args, **kwargs)

    def heavy_residency_refresh(*args, **kwargs):
        refreshes.append("browser-heavy-residency")
        return original_heavy_residency(*args, **kwargs)

    def scientific_refresh(*args, **kwargs):
        refreshes.append("scientific")
        return original_scientific(*args, **kwargs)

    monkeypatch.setattr(controls, "reconcile", controls_reconcile)
    monkeypatch.setattr(page._shell.browser, "reconcile", browser_refresh)
    monkeypatch.setattr(
        page._shell.browser,
        "reconcile_heavy_residency",
        heavy_residency_refresh,
    )
    monkeypatch.setattr(
        page._shell.scientific, "reconcile", scientific_refresh,
    )
    projection = controls.projection
    assert projection is not None
    initial_fields = tuple(
        (field.path, field.enabled)
        for field in projection.fields
    )
    assert any(enabled for _path, enabled in initial_fields)

    try:
        page._shell.browser.metadata.click()
        qapp.processEvents()
        dialog = page._metadata_dialog
        assert dialog is not None and dialog.isVisible()
        assert metadata_entered.wait(2)
        assert refreshes == ["controls"]
        projection = controls.projection
        assert projection is not None
        assert tuple(
            (field.path, field.enabled)
            for field in projection.fields
        ) == initial_fields
        assert not page._shell.scientific.background.isEnabled()
        active_actions = {
            action.action.value: action.enabled
            for actions in projection.section_actions.values()
            for action in actions
        }
        assert not active_actions["calibrate"]
        assert not active_actions["make_mask"]
        assert not active_actions["reintegrate_1d"]
        assert not active_actions["reintegrate_2d"]

        page._drain_executor()
        qapp.processEvents()
        assert page._metadata_dialog is dialog and dialog.isVisible()
        assert dialog.status.text() == "Analysis: metadata 1/2"
        assert refreshes == ["controls"]

        metadata_release.set()
        worker = page._analysis_slot._worker
        assert worker is not None
        worker.join(2)
        assert not worker.is_alive()
        page._drain_executor()
        qapp.processEvents()
        assert requalification_entered.wait(2)
        assert page._metadata_dialog is dialog and dialog.isVisible()
        assert refreshes == ["controls", "controls"]

        page._drain_executor()
        qapp.processEvents()
        assert dialog.status.text() == (
            "Analysis: metadata_requalification 1/2"
        )
        assert refreshes == ["controls", "controls"]

        requalification_release.set()
        worker = page._analysis_slot._worker
        assert worker is not None
        worker.join(2)
        assert not worker.is_alive()
        page._drain_executor()
        qapp.processEvents()
        assert page._metadata_dialog is dialog and dialog.isVisible()
        assert page._metadata_result is table
        assert dialog.table.rowCount() == len(table.labels)
        assert dialog.status.text() == "OK"
        assert refreshes == ["controls", "controls", "controls"]
        projection = controls.projection
        assert projection is not None
        assert tuple(
            (field.path, field.enabled)
            for field in projection.fields
        ) == initial_fields
        assert page._shell.scientific.background.isEnabled()
    finally:
        metadata_release.set()
        requalification_release.set()
        _close_page(page, qapp)


def test_metadata_analysis_owner_accepts_a_real_control_edit(
    monkeypatch, qapp,
) -> None:
    import xrd_tools.analysis.scan_operations as scan_operations
    from xdart.gui.tabs.scattering.analysis_mount import analysis_request_facts

    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(processing_mode="2D Viewer")),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    entered, release = Event(), Event()
    plan = MetadataTablePlan("/tmp/current.nexus")
    facts = analysis_request_facts(plan)

    def run_metadata(actual, *, cancel_token, progress_callback):
        assert actual == plan
        entered.set()
        assert release.wait(2)
        return _table("/tmp/current.nexus", fingerprint="edited-stale")

    monkeypatch.setattr(
        scan_operations, "run_metadata_table", run_metadata,
    )
    monkeypatch.setattr(
        page, "_current_analysis_request", lambda kind, target: facts,
    )
    try:
        page._open_analysis_mount("metadata")
        identity = page._submit_metadata(
            plan, target="metadata", request=facts,
        )
        assert type(identity) is OperationIdentity
        assert entered.wait(2)
        assert page._start_permitted() == (
            False,
            "Analysis operation is still active",
        )
        assert not page._shell.run_controls.startButton.isEnabled()
        assert page._shell.run_controls.readinessLabel.text() == (
            "Analysis operation is still active"
        )

        invoked = []
        monkeypatch.setattr(
            page, "_run_action", lambda: invoked.append("run"),
        )
        monkeypatch.setattr(
            page, "_background_action", lambda: invoked.append("background"),
        )
        monkeypatch.setattr(
            page, "_calibrate_action", lambda: invoked.append("calibrate"),
        )
        monkeypatch.setattr(
            page, "_mask_action", lambda: invoked.append("mask"),
        )
        monkeypatch.setattr(
            page, "_reintegrate_action",
            lambda dimension: invoked.append(f"reintegrate_{dimension}"),
        )
        for command in (
            ShellCommand(ShellCommandKind.RUN_ACTION),
            ShellCommand(ShellCommandKind.SET_BACKGROUND),
            ShellCommand(ShellCommandKind.CONTROL_ACTION, "calibrate"),
            ShellCommand(ShellCommandKind.CONTROL_ACTION, "make_mask"),
            ShellCommand(ShellCommandKind.CONTROL_ACTION, "reintegrate_1d"),
            ShellCommand(ShellCommandKind.CONTROL_ACTION, "reintegrate_2d"),
        ):
            page._handle_shell_command(command)
        assert invoked == []
        assert page._notice_text == "Analysis operation is still active."

        before = page._intents.snapshot()
        next_cores = 1 if before.thaw().max_cores != 1 else 2
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SET_CORES, next_cores,
        ))
        after = page._intents.snapshot()
        assert after.revision == before.revision + 1
        assert after.thaw().max_cores == next_cores
        assert page._analysis_slot.current_identity is identity
    finally:
        release.set()
        worker = page._analysis_slot._worker
        if worker is not None:
            worker.join(2)
        page._drain_executor()
        _close_page(page, qapp)


def test_pending_reintegrate_reload_uses_one_mutating_busy_truth(
    monkeypatch, qapp,
) -> None:
    import xdart.gui.tabs.scattering.page as page_module
    from xdart.gui.tabs.scattering.analysis_mount import (
        analysis_request_facts,
        analysis_start_allowed,
    )

    page = _page()
    controls = page._shell.controls
    monkeypatch.setattr(
        page_module, "resolve_calibration_executable", lambda: "/tmp/calib2",
    )
    monkeypatch.setattr(
        page_module, "resolve_mask_executable", lambda: "/tmp/drawmask",
    )
    monkeypatch.setattr(
        page,
        "_capture_current_loaded_browse",
        lambda: object(),
    )

    refreshes = []
    controls_reconcile = controls.reconcile
    browser_reconcile = page._shell.browser.reconcile
    scientific_reconcile = page._shell.scientific.reconcile

    def reconcile_controls(state):
        refreshes.append("controls")
        return controls_reconcile(state)

    def reconcile_browser(*args, **kwargs):
        refreshes.append("browser")
        return browser_reconcile(*args, **kwargs)

    def reconcile_scientific(*args, **kwargs):
        refreshes.append("scientific")
        return scientific_reconcile(*args, **kwargs)

    monkeypatch.setattr(controls, "reconcile", reconcile_controls)
    monkeypatch.setattr(page._shell.browser, "reconcile", reconcile_browser)
    monkeypatch.setattr(
        page._shell.scientific, "reconcile", reconcile_scientific,
    )

    plan = MetadataTablePlan("/tmp/current.nexus")
    request = analysis_request_facts(plan)
    invoked = []
    monkeypatch.setattr(page, "_run_action", lambda: invoked.append("run"))
    monkeypatch.setattr(
        page, "_calibrate_action", lambda: invoked.append("calibrate"),
    )
    monkeypatch.setattr(
        page, "_mask_action", lambda: invoked.append("make_mask"),
    )
    monkeypatch.setattr(
        page, "_reintegrate_action",
        lambda dimension: invoked.append(f"reintegrate_{dimension}"),
    )

    try:
        page._open_analysis_mount("metadata")
        assert not page._workspace_operations.owned
        _set_pending_reintegrate_reload(page)
        assert page._experiment_operation_busy()
        assert not analysis_start_allowed(page)
        assert page._submit_metadata(
            plan,
            target="metadata",
            request=request,
        ) is None

        page._refresh_shell(
            preserve_display=True,
            preserve_scientific=True,
        )
        projection = controls.projection
        assert projection is not None
        action_enabled = {
            action.action.value: action.enabled
            for actions in projection.section_actions.values()
            for action in actions
        }
        assert not action_enabled["calibrate"]
        assert not action_enabled["make_mask"]
        assert not action_enabled["reintegrate_1d"]
        assert not action_enabled["reintegrate_2d"]
        assert not page._shell.scientific.background.isEnabled()
        assert refreshes == ["controls"]

        for command in (
            ShellCommand(ShellCommandKind.RUN_ACTION),
            ShellCommand(ShellCommandKind.SET_BACKGROUND),
            ShellCommand(ShellCommandKind.CONTROL_ACTION, "calibrate"),
            ShellCommand(ShellCommandKind.CONTROL_ACTION, "make_mask"),
            ShellCommand(ShellCommandKind.CONTROL_ACTION, "reintegrate_1d"),
            ShellCommand(ShellCommandKind.CONTROL_ACTION, "reintegrate_2d"),
        ):
            page._handle_shell_command(command)
        assert invoked == []
        assert set(refreshes) == {"controls"}
        assert "controls-rebuild" not in refreshes
        assert "browser" not in refreshes
        assert "scientific" not in refreshes

        _clear_pending_reintegrate_reload(page)
        page._refresh_shell(
            preserve_display=True,
            preserve_scientific=True,
        )
        assert analysis_start_allowed(page)
        projection = controls.projection
        assert projection is not None
        action_enabled = {
            action.action.value: action.enabled
            for actions in projection.section_actions.values()
            for action in actions
        }
        assert action_enabled["calibrate"]
        assert action_enabled["make_mask"]
        assert action_enabled["reintegrate_1d"]
        assert action_enabled["reintegrate_2d"]
        assert page._shell.scientific.background.isEnabled()
    finally:
        _clear_pending_reintegrate_reload(page)
        _close_page(page, qapp)


def test_pending_reintegrate_reload_dominates_active_background_mutation(
    monkeypatch, qapp,
) -> None:
    page = _page()
    original_refresh = page._refresh_shell
    refreshes = []
    releases = []
    science_reconciles = []
    background_owner = SimpleNamespace(
        phase="ACTIVE",
        active_key=None,
        projection=lambda: None,
    )

    def refresh(*args, **kwargs):
        refreshes.append((args, kwargs))
        return original_refresh(*args, **kwargs)

    scientific_reconcile = page._shell.scientific.reconcile

    def reconcile_scientific(*args, **kwargs):
        science_reconciles.append((args, kwargs))
        return scientific_reconcile(*args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(page, "_background_owner", background_owner)
            patch.setattr(
                page,
                "_release_display_background",
                lambda: releases.append(True) or True,
            )
            patch.setattr(page, "_refresh_shell", refresh)
            patch.setattr(
                page._shell.scientific,
                "reconcile",
                reconcile_scientific,
            )

            _set_pending_reintegrate_reload(page)
            assert not page._workspace_operations.owned

            page._handle_shell_command(
                ShellCommand(ShellCommandKind.SET_BACKGROUND)
            )
            page._background_action()

            assert releases == []
            assert science_reconciles == []
            assert refreshes == [
                ((), {
                    "preserve_display": True,
                    "preserve_scientific": True,
                }),
                ((), {
                    "preserve_display": True,
                    "preserve_scientific": True,
                }),
            ]

            _clear_pending_reintegrate_reload(page)
            page._handle_shell_command(
                ShellCommand(ShellCommandKind.SET_BACKGROUND)
            )
            assert releases == [True]

            identity = OperationIdentity(901)
            active_slot = SimpleNamespace(
                owned=True,
                current_identity=identity,
                observe_stamp=lambda _stamp: None,
            )
            _set_pending_reintegrate_reload(page)
            page._background_identity = identity
            patch.setattr(page._workspace_operations, "_slot", active_slot)
            page._handle_shell_command(
                ShellCommand(ShellCommandKind.SET_BACKGROUND)
            )
            assert releases == [True, True]
    finally:
        _clear_pending_reintegrate_reload(page)
        page._background_identity = None
        _close_page(page, qapp)


def test_raw_tiff_viewer_metadata_click_uses_auto_sidecar_plan(
    monkeypatch, qapp, tmp_path,
) -> None:
    tifffile = pytest.importorskip("tifffile")
    image = tmp_path / "scan_0001.TIFF"
    tifffile.imwrite(image, np.ones((3, 4), dtype=np.uint16))
    image.with_suffix(".txt").write_text(
        "# Counters\nI0 = 10\n"
        "# Motors\nth = 0.1\n"
        "User: p37, time: Mon Jan 15 10:30:00 2024  # Temp\n",
        encoding="utf-8",
    )
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(processing_mode="2D Viewer")),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    captured_plans = []
    begin_metadata = page._analysis_slot.begin_metadata

    def begin(plan, stamp):
        captured_plans.append(plan)
        return begin_metadata(plan, stamp)

    monkeypatch.setattr(page._analysis_slot, "begin_metadata", begin)

    def wait_for(predicate) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            qapp.processEvents()
            page._drain_executor()
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError("TIFF metadata operation did not settle")

    try:
        page._open_viewer_2d_path(str(image))
        wait_for(lambda: page._context_controller.viewer_2d_frame is not None)
        current = page._context_controller.navigation.current
        assert current is not None
        assert current.source_scan == current.artifact == "viewer-2d"
        assert page._context_controller.viewer_2d_context.original_path == str(
            image
        )
        with monkeypatch.context() as patch:
            patch.setattr(
                page._context_controller,
                "owns_frame",
                lambda _frame: False,
            )
            assert page._metadata_plan_for_current() is None
        owner = page._context_controller._viewer_2d
        context = owner.context
        assert context is not None
        with monkeypatch.context() as patch:
            patch.setattr(
                owner,
                "context",
                replace(
                    context,
                    original_path=str(tmp_path / "stale.TIFF"),
                ),
            )
            assert page._metadata_plan_for_current() is None

        page._shell.browser.metadata.click()
        qapp.processEvents()

        assert len(captured_plans) == 1
        plan = captured_plans[0]
        assert plan.source == str(image)
        assert plan.metadata_format == "auto"
        from xdart.gui.tabs.scattering.analysis_mount import (
            analysis_request_facts,
        )
        request = analysis_request_facts(plan)
        assert request is not None
        assert request[1][-1] == "auto"
        process = page._metadata_operations.active
        assert process is not None
        assert process.request.request == request
        wait_for(lambda: page._metadata_result is not None)

        result = page._metadata_result
        assert result is not None
        columns = {column.name: column for column in result.columns}
        np.testing.assert_allclose(columns["I0"].numeric, [10.0])
        np.testing.assert_allclose(columns["th"].numeric, [0.1])
        assert page._metadata_dialog is not None
        headers = tuple(
            page._metadata_dialog.table.horizontalHeaderItem(index).text()
            for index in range(page._metadata_dialog.table.columnCount())
        )
        assert "I0" in headers and "th" in headers
    finally:
        _close_page(page, qapp)

    from tests.xdart.scattering.test_e3_context_contract import _browse
    from xdart.modules.display_context import ContextKind, new_context_token

    processed = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(processing_mode="2D Viewer")),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    request, browse = _browse(
        new_context_token(ContextKind.BROWSE),
        1,
        scan_key="processed-metadata",
    )
    try:
        processed._context_controller._runtime.adopt_browse(browse, request)
        processed_plan = processed._metadata_plan_for_current()
        assert processed_plan is not None
        assert str(processed_plan.source).endswith(".nxs")
        assert processed_plan.metadata_format is None
        processed_request = processed._current_analysis_request(
            "metadata", "metadata",
        )
        assert processed_request is not None
        assert processed_request[1][-1] is None
    finally:
        _close_page(processed, qapp)


@pytest.mark.parametrize("stale", ("dialog", "generation", "request", "denied"))
def test_deferred_metadata_permanent_or_stale_state_drops_without_launch(
    stale, monkeypatch, qapp,
) -> None:
    from xdart.gui.tabs.scattering.analysis_mount import (
        analysis_request_facts,
    )

    page = _page()
    original_slot = page._analysis_slot
    original_dialog = None
    try:
        page._open_analysis_mount("metadata")
        original_dialog = page._metadata_dialog
        plan = MetadataTablePlan("/tmp/exact.nxs")
        exact = analysis_request_facts(plan)
        changed = analysis_request_facts(MetadataTablePlan("/tmp/changed.nxs"))
        current = [exact]
        launches = []
        slot = SimpleNamespace(
            owned=True,
            begin_metadata=lambda candidate, stamp:
                launches.append((candidate, stamp)) or OperationIdentity(46),
        )
        page._analysis_slot = slot
        monkeypatch.setattr(
            page, "_current_analysis_request",
            lambda _kind, _target: current[0],
        )
        assert page._submit_metadata(
            plan, request=exact,
        ) is None
        deferred = page._metadata_operations.deferred
        assert deferred is not None

        slot.owned = False
        if stale == "dialog":
            page._metadata_operations.close_dialog(deferred.dialog)
        elif stale == "generation":
            page._metadata_operations.close_dialog(deferred.dialog)
            assert page._metadata_operations.open_dialog("metadata") is not None
        elif stale == "request":
            current[0] = changed
        else:
            page._admission_state = object()

        effect = page._dispatch_deferred_metadata()
        assert effect is (
            WorkspaceRefreshEffect.NONE
            if stale in {"dialog", "generation"}
            else WorkspaceRefreshEffect.CONTROLS
        )
        assert page._metadata_operations.deferred is None
        assert launches == []
    finally:
        page._analysis_slot = original_slot
        page._admission_state = None
        if original_dialog is not None:
            page._metadata_dialog = original_dialog
        _close_page(page, qapp)


def test_busy_analysis_slot_keeps_only_latest_metadata_request(
    monkeypatch, qapp,
) -> None:
    from xdart.gui.tabs.scattering.analysis_mount import analysis_request_facts

    page = _page()
    original_slot = page._analysis_slot
    try:
        page._open_analysis_mount("metadata")
        first = MetadataTablePlan("/tmp/first.nxs")
        latest = MetadataTablePlan("/tmp/latest.nxs")
        first_facts = analysis_request_facts(first)
        latest_facts = analysis_request_facts(latest)
        launches = []
        identity = OperationIdentity(46)
        slot = SimpleNamespace(
            owned=True,
            begin_metadata=lambda plan, stamp:
                launches.append((plan, stamp)) or identity,
        )
        page._analysis_slot = slot
        current = [first_facts]
        monkeypatch.setattr(
            page, "_current_analysis_request",
            lambda _kind, _target: current[0],
        )

        assert page._submit_metadata(
            first, request=first_facts,
        ) is None
        current[0] = latest_facts
        assert page._submit_metadata(
            latest, request=latest_facts,
        ) is None
        assert page._metadata_operations.deferred is not None
        assert page._metadata_operations.deferred.plan is latest
        assert launches == []

        slot.owned = False
        assert page._dispatch_deferred_metadata()
        assert launches and launches[0][0] is latest
        assert page._metadata_operations.deferred is None
        assert page._metadata_operations.active_identity is identity
    finally:
        page._analysis_slot = original_slot
        _close_page(page, qapp)


def test_metadata_lost_owner_preserves_newer_request_and_stale_token_cannot_paint(
    qapp,
) -> None:
    from xdart.gui.tabs.scattering.analysis_mount import analysis_request_facts

    page = _page()
    try:
        page._open_analysis_mount("metadata")
        page._open_analysis_mount("scan_roi")
        original_dialog = page._metadata_dialog
        original_token = page._metadata_operations.dialog_identity("metadata")
        assert original_dialog is not None and original_token is not None

        old_plan = MetadataTablePlan("/tmp/lost-owner.nxs")
        old_facts = analysis_request_facts(old_plan)
        identity = OperationIdentity(47)
        process = _start_metadata_process(
            page, identity, old_plan, old_facts,
        )
        latest_plan = MetadataTablePlan("/tmp/latest-owner.nxs")
        latest_facts = analysis_request_facts(latest_plan)
        latest_dialog = page._metadata_operations.dialog_identity("scan_roi")
        latest = page._metadata_operations.capture(
            latest_plan,
            latest_facts,
            target="scan_roi",
            dialog=latest_dialog,
        )
        assert latest is not None

        assert page._metadata_operations.lost(identity) is process
        assert page._metadata_operations.deferred is latest
        assert page._finish_metadata_refresh(
            original_token,
            "Analysis failed before terminal publication.",
        ) is WorkspaceRefreshEffect.DIALOG

        original_dialog.close()
        qapp.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(
            None, QtCore.QEvent.Type.DeferredDelete,
        )
        page._open_analysis_mount("metadata")
        reopened = page._metadata_dialog
        assert reopened is not None and reopened is not original_dialog
        assert (
            page._metadata_operations.dialog_identity("metadata")
            is not original_token
        )
        before = reopened.status.text()
        page._set_metadata_dialog_status(original_token, "stale terminal")
        assert reopened.status.text() == before
        assert page._metadata_operations.deferred is latest
    finally:
        _close_page(page, qapp)


def test_lost_metadata_owner_handoff_reconciles_controls_once(
    monkeypatch, qapp,
) -> None:
    from xdart.gui.tabs.scattering.analysis_mount import analysis_request_facts

    page = _page()
    original_slot = page._analysis_slot
    latest_identity = OperationIdentity(902)
    try:
        page._open_analysis_mount("metadata")
        page._open_analysis_mount("scan_roi")
        old_plan = MetadataTablePlan("/tmp/aborted-owner.nxs")
        latest_plan = MetadataTablePlan("/tmp/replacement-owner.nxs")
        old_facts = analysis_request_facts(old_plan)
        latest_facts = analysis_request_facts(latest_plan)
        old_identity = OperationIdentity(901)
        _start_metadata_process(
            page, old_identity, old_plan, old_facts,
        )
        latest_dialog = page._metadata_operations.dialog_identity("scan_roi")
        latest = page._metadata_operations.capture(
            latest_plan,
            latest_facts,
            target="scan_roi",
            dialog=latest_dialog,
        )
        assert latest is not None

        launches = []
        slot = SimpleNamespace(
            current_identity=old_identity,
            owned=True,
            observe_stamp=lambda _stamp: None,
            cancel=lambda _identity: False,
        )

        def poll(identity):
            assert identity is old_identity
            slot.current_identity = None
            slot.owned = False
            return None

        def begin_metadata(plan, stamp):
            launches.append((plan, stamp))
            slot.current_identity = latest_identity
            slot.owned = True
            return latest_identity

        slot.poll = poll
        slot.begin_metadata = begin_metadata
        page._analysis_slot = slot
        monkeypatch.setattr(
            page,
            "_current_analysis_request",
            lambda _kind, target:
                old_facts if target == "metadata" else latest_facts,
        )
        refreshes = []
        monkeypatch.setattr(
            page,
            "_refresh_shell",
            lambda **kwargs: refreshes.append(kwargs),
        )

        page._drain_executor()

        assert launches == [(latest_plan, page._metadata_context_stamp())]
        assert page._metadata_operations.deferred is None
        assert page._metadata_operations.active_identity is latest_identity
        assert refreshes == [{
            "preserve_display": True,
            "preserve_scientific": True,
        }]
    finally:
        page._metadata_operations.finish(latest_identity)
        page._analysis_slot = original_slot
        _close_page(page, qapp)


def test_newer_cross_target_metadata_request_preempts_old_requalification(
    monkeypatch, qapp,
) -> None:
    from xdart.gui.tabs.scattering.analysis_mount import analysis_request_facts

    page = _page()
    original_slot = page._analysis_slot
    try:
        page._open_analysis_mount("metadata")
        page._open_analysis_mount("scan_roi")
        first = MetadataTablePlan("/tmp/first.nxs")
        latest = MetadataTablePlan("/tmp/latest.nxs")
        first_facts = analysis_request_facts(first)
        latest_facts = analysis_request_facts(latest)
        identities = iter((OperationIdentity(47), OperationIdentity(48)))
        launches = []
        slot = SimpleNamespace(
            owned=False,
            begin_metadata=lambda plan, stamp:
                launches.append(("metadata", plan, stamp)) or next(identities),
            begin_metadata_requalification=lambda plan, stamp:
                launches.append(("requalification", plan, stamp))
                or pytest.fail("older result displaced newer request"),
        )
        page._analysis_slot = slot

        def current(kind, target):
            if kind in {"metadata", "metadata_requalification"}:
                return first_facts if target == "metadata" else latest_facts
            return None

        monkeypatch.setattr(page, "_current_analysis_request", current)
        first_identity = page._submit_metadata(
            first, target="metadata", request=first_facts,
        )
        assert first_identity == OperationIdentity(47)
        slot.owned = True
        assert page._submit_metadata(
            target="scan_roi", request=latest_facts,
            plan=latest,
        ) is None
        assert page._metadata_operations.deferred.plan is latest
        old_status = page._metadata_dialog.status.text()
        assert page._notice_text == "Metadata queued…"
        assert page._consume_analysis_update(OperationUpdate(
            first_identity,
            progress=OperationProgress(
                first_identity, "metadata", 1, 2, 1,
            ),
        )) is WorkspaceRefreshEffect.DIALOG
        assert page._notice_text == "Metadata queued…"
        assert page._metadata_dialog.status.text() == old_status

        slot.owned = False
        assert page._consume_analysis_update(
            _returned(first_identity, _table("/tmp/first.nxs"))
        )
        assert page._metadata_operations.deferred.plan is latest
        assert [kind for kind, _plan, _stamp in launches] == ["metadata"]
        assert page._dispatch_deferred_metadata()
        assert launches[-1][0] == "metadata"
        assert launches[-1][1] is latest
    finally:
        page._analysis_slot = original_slot
        _close_page(page, qapp)


def test_newer_cross_target_metadata_request_suppresses_old_requalified_result(
    monkeypatch, qapp,
) -> None:
    from xdart.gui.tabs.scattering.analysis_mount import analysis_request_facts

    page = _page()
    original_slot = page._analysis_slot
    try:
        page._open_analysis_mount("metadata")
        page._open_analysis_mount("scan_roi")
        old_plan = MetadataTablePlan("/tmp/old.nxs")
        latest = MetadataTablePlan("/tmp/latest.nxs")
        old_facts = analysis_request_facts(old_plan)
        latest_facts = analysis_request_facts(latest)
        old_candidate = _table("/tmp/old.nxs", fingerprint="old-table")
        old_identity = OperationIdentity(49)
        latest_identity = OperationIdentity(50)
        launches = []
        slot = SimpleNamespace(
            owned=False,
            begin_metadata=lambda plan, stamp:
                launches.append(("metadata", plan, stamp)) or latest_identity,
            begin_metadata_requalification=lambda plan, stamp:
                launches.append(("requalification", plan, stamp))
                or old_identity,
        )
        page._analysis_slot = slot

        def current(kind, target):
            if kind in {"metadata", "metadata_requalification"}:
                return old_facts if target == "metadata" else latest_facts
            return None

        monkeypatch.setattr(page, "_current_analysis_request", current)
        assert page._submit_metadata(
            None,
            target="metadata",
            request=old_facts,
            candidate=old_candidate,
        ) == old_identity
        slot.owned = True
        assert page._submit_metadata(
            latest, target="scan_roi", request=latest_facts,
        ) is None
        assert page._metadata_operations.deferred.plan is latest

        slot.owned = False
        validation = MetadataTableRequalificationResult(
            AnalysisDisposition.COMPLETED, "OK", old_candidate.receipt,
            old_candidate.table_fingerprint,
        )
        assert page._consume_analysis_update(
            _returned(old_identity, validation)
        )
        assert page._metadata_result is None
        assert page._metadata_dialog.table.rowCount() == 0
        assert page._metadata_operations.deferred.plan is latest

        assert page._dispatch_deferred_metadata()
        assert launches[-1][0] == "metadata"
        assert launches[-1][1] is latest
        assert page._metadata_operations.active_identity == latest_identity
        assert page._metadata_operations.deferred is None
    finally:
        page._analysis_slot = original_slot
        _close_page(page, qapp)


def test_metadata_reuse_is_pure_and_worker_requalification_rejects_replacement(
        tmp_path) -> None:
    import xdart.gui.tabs.scattering.analysis_mount as mount

    artifact = tmp_path / "scan.nxs"
    artifact.write_bytes(b"first revision")
    source = SourceSpec(
        artifact, SourceKind.PROCESSED_NEXUS, entry="entry",
        options={"scan": "7"},
    )
    table = _table(source, primary_post_state=_file_revision(artifact))
    facts = mount.analysis_request_facts(MetadataTablePlan(source))

    assert mount.analysis_result_matches(table, facts)
    changed_options = replace(source, options={"scan": "8"})
    changed_facts = mount.analysis_request_facts(MetadataTablePlan(changed_options))
    assert not mount.analysis_result_matches(table, changed_facts)

    artifact.write_bytes(b"a distinct replacement revision")
    assert mount.analysis_result_matches(table, facts)
    validation = run_metadata_table_requalification(
        MetadataTableRequalificationPlan(
            table.receipt, table.table_fingerprint,
        )
    )
    assert validation.disposition is AnalysisDisposition.REFUSED
    assert validation.code == "SOURCE_REVISION_CHANGED"
    assert not mount.analysis_result_matches(
        replace(table, receipt=replace(table.receipt, primary_post_state=None)),
        facts,
    )


def test_p37b_source_widget_external_mode_does_zero_discovery_probe_or_io(
        monkeypatch, qapp) -> None:
    from xdart.gui.analysis.scan_source_widget import ScanSourceWidget
    import xrd_tools.analysis.scan_operations as scan

    def forbidden(*_args, **_kwargs):
        pytest.fail("vNext source projection performed GUI discovery/probe/I/O")

    widget = ScanSourceWidget(mode="vnext_analysis")
    try:
        assert widget._external_execution and widget._probe_executor is None
        widget.dir_check.setChecked(True)
        widget.path_edit.setText("/tmp/real-folder-syntax")
        folder = widget.external_source_syntax()
        assert folder[0:2] == ("directory", "/tmp/real-folder-syntax")
        assert type(folder[2]) is str and folder[2]
        monkeypatch.setattr(widget, "_refresh_candidates", forbidden)
        monkeypatch.setattr(widget, "_start_async_probe", forbidden)
        monkeypatch.setattr(ScanSourceWidget, "_probe_source", staticmethod(forbidden))
        monkeypatch.setattr(ScanSourceWidget, "_file_candidates", staticmethod(forbidden))
        monkeypatch.setattr(scan, "discover_scans", forbidden)
        first = SourceSpec("/tmp/a.nxs", SourceKind.PROCESSED_NEXUS)
        second = SourceSpec("/tmp/b.nxs", SourceKind.PROCESSED_NEXUS)
        widget.set_external_candidates((
            CandidateProjection(first, "a-fp", 10),
            CandidateProjection(second, "b-fp", 11),
        ))
        assert widget.scan_combo.count() == 3
        assert widget.scan_combo.currentData() is None
        assert widget.scan_combo.currentText() == "Choose a scan…"
        assert widget.external_source_syntax() is None
        widget.scan_combo.setCurrentIndex(2)
        assert widget.external_source_syntax() == ("exact", second)
        assert widget._probe_executor is None and widget._last_selection is None
    finally:
        widget.close(); qapp.processEvents()


def test_p37b_scan_plot_submits_headless_defaults_normalization_and_roi_kind(
        monkeypatch, qapp) -> None:
    page = _page()
    try:
        page._open_analysis_mount("scan_roi")
        dialog = page._scan_roi_dialog
        table = _table(fingerprint="scan-table")
        dialog.set_vnext_metadata(table)
        dialog._vnext_rendering = True
        try:
            _check_item(dialog.y_list, "signal")
            _check_item(dialog.r_list, "motor")
            dialog.norm_combo.setCurrentText("signal")
            dialog.log_btn.setChecked(True)
        finally:
            dialog._vnext_rendering = False
        launches = []
        identity = OperationIdentity(51)
        monkeypatch.setattr(page._analysis_slot, "begin_scan_plot",
            lambda plan, actual_table, roi, stamp:
                launches.append((plan, actual_table, roi)) or identity)
        values = dialog.vnext_plot_values()
        page._scan_analysis_action("scan_plot", values)
        plan, actual_table, roi = launches[-1]
        assert actual_table is table and roi is None
        assert (plan.x, plan.y, plan.normalization) == (
            "motor", ("signal", "motor"), "signal")
        result = ScanPlotResult(
            AnalysisDisposition.COMPLETED, "OK",
            table_fingerprint=table.table_fingerprint, x_name="motor",
            x=np.array([10.0, 11.0, 12.0]),
            trace_names=("signal", "motor"),
            trace_origins=("metadata", "metadata"),
            original_identities=("signal", "motor"),
            traces=(np.array([0.4, 0.45, 0.5]), np.ones(3)),
            normalization="signal", storage_bytes=144)
        assert page._consume_analysis_update(_returned(identity, result))
        assert page._scan_roi_result is result
        assert dialog._vnext_scan_result is result
        assert dialog._vnext_scan_request == (
            table.table_fingerprint, None, "motor",
            ("signal", "motor"), "signal")
        assert dialog.right_axis.isVisible() and dialog.log_btn.isChecked()

        dialog._table = {}; dialog._columns = []
        before = len(launches)
        dialog._vnext_rendering = True
        try:
            _check_item(dialog.y_list, "motor")
        finally:
            dialog._vnext_rendering = False
        assert dialog.vnext_plot_values()[1] == ("signal", "motor")
        _check_item(dialog.r_list, "motor", checked=False)
        dialog.log_btn.setChecked(False)
        assert len(launches) == before
        assert not dialog.right_axis.isVisible()
        assert len(dialog.plot.listDataItems()) == 2

        dialog._vnext_rendering = True
        try:
            _check_item(dialog.y_list, "signal", checked=False)
            _check_item(dialog.y_list, "motor", checked=False)
        finally:
            dialog._vnext_rendering = False
        dialog._redraw()
        assert len(launches) == before + 1
        assert launches[-1][0].y == ()

        roi_result = RoiScanResult(
            AnalysisDisposition.COMPLETED, "OK", receipt=table.receipt,
            table_fingerprint=table.table_fingerprint,
            requested_labels=table.labels, completed_labels=table.labels,
            signal_names=("roi_sum",), signal_values=(np.ones(3),),
            valid_counts=(np.ones(3, dtype=int),), result_fingerprint="roi-fp")
        dialog.set_vnext_roi_result(roi_result)
        assert len(dialog.y_list.findItems(
            "roi_sum", QtCore.Qt.MatchFlag.MatchExactly)) == 1
    finally:
        _close_page(page, qapp)


def test_p37b_roi_preview_and_scan_share_slot_and_preserve_typed_diagnostics(
        monkeypatch, qapp) -> None:
    page = _page()
    try:
        page._open_analysis_mount("scan_roi")
        dialog = page._scan_roi_dialog
        first = _table(fingerprint="roi-table-1", source_fingerprint="source-1")
        dialog.set_vnext_metadata(first)
        preview_plans = []
        preview_identities = iter((OperationIdentity(61), OperationIdentity(62)))
        monkeypatch.setattr(page._analysis_slot, "begin_roi_preview",
            lambda plan, stamp: preview_plans.append(plan) or next(preview_identities))
        roi_plans = []
        roi_identity = OperationIdentity(63)
        monkeypatch.setattr(page._analysis_slot, "begin_roi_scan",
            lambda plan, stamp: roi_plans.append(plan) or roi_identity)

        dialog.roi_btn.click()
        first_identity = page._analysis_identity
        first_preview = RoiPreviewResult(
            AnalysisDisposition.COMPLETED, "OK", receipt=first.receipt,
            table_fingerprint=first.table_fingerprint, labels=first.labels,
            label=first.labels[0], image=np.arange(16.0).reshape(4, 4),
            result_fingerprint="preview-1", storage_bytes=128)
        assert page._consume_analysis_update(_returned(first_identity, first_preview))
        old_picker = dialog._roi_dialog
        old_signals = tuple(old_picker.roi_signals())
        assert old_picker is not None and page._roi_preview_binding[-1] is old_picker
        assert page._roi_preview_binding[0] is first
        assert page._roi_preview_binding[1:6] == (
            first.receipt, first.table_fingerprint, "preview-1",
            first.labels[0], first.labels)

        second = _table("/tmp/second.nxs", fingerprint="roi-table-2",
                        source_fingerprint="source-2")
        dialog.set_vnext_metadata(second)
        page._roi_analysis_signals(first, old_picker, old_signals)
        assert not roi_plans

        dialog.roi_btn.click()
        second_identity = page._analysis_identity
        second_preview = RoiPreviewResult(
            AnalysisDisposition.COMPLETED, "OK", receipt=second.receipt,
            table_fingerprint=second.table_fingerprint, labels=second.labels,
            label=second.labels[0], image=np.arange(25.0).reshape(5, 5),
            result_fingerprint="preview-2", storage_bytes=200)
        assert page._consume_analysis_update(_returned(second_identity, second_preview))
        picker = dialog._roi_dialog
        assert page._roi_preview_binding[0] is second
        assert page._roi_preview_binding[1:6] == (
            second.receipt, second.table_fingerprint, "preview-2",
            second.labels[0], second.labels)
        assert page._roi_preview_binding[-1] is picker
        picker.mask_sat_check.setChecked(True)
        signals = tuple(picker.roi_signals())
        page._roi_analysis_signals(second, picker, signals)
        assert page._analysis_identity is roi_identity and len(roi_plans) == 1
        plan = roi_plans[0]
        assert plan.receipt == second.receipt
        assert plan.table_fingerprint == second.table_fingerprint
        assert plan.signals == signals and plan.mask is None
        assert plan.mask_saturation is True
        completed = RoiScanResult(
            AnalysisDisposition.COMPLETED, "OK",
            diagnostics=("ROI_INVALID_MASK_IGNORED",), receipt=second.receipt,
            table_fingerprint=second.table_fingerprint,
            requested_labels=second.labels, completed_labels=second.labels,
            signal_names=tuple(signal.name for signal in signals),
            signal_values=tuple(np.ones(3) for _ in signals),
            valid_counts=tuple(np.ones(3, dtype=int) for _ in signals),
            policy_fingerprint="roi-policy", result_fingerprint="roi-result",
            storage_bytes=96)
        assert page._consume_analysis_update(_returned(roi_identity, completed))
        assert page._scan_roi_result is completed
        assert page._scan_roi_result.diagnostics == (
            "ROI_INVALID_MASK_IGNORED",)
        assert dialog.status.text() == "OK"
        for name in completed.signal_names:
            assert dialog.y_list.findItems(name, QtCore.Qt.MatchFlag.MatchExactly)
    finally:
        _close_page(page, qapp)


def test_p37b_peak_uses_exact_current_trace_and_headless_selection_policy(
        monkeypatch, qapp) -> None:
    import xdart.gui.tabs.scattering.analysis_mount as mount

    for epoch in (None, 1700000000.25, -0.0):
        current, projection = _projection(epoch=epoch)
        trace = mount.displayed_trace_input(projection, current)
        expected = (0 if epoch is None else
                    int.from_bytes(struct.pack(">d", epoch), "big") + 1)
        assert trace.epoch == expected == mount.epoch_token(epoch)
        assert trace.title == "current" and trace.receipt.frame_label == "3"
        assert trace.receipt.work_ordinal == 9
        if epoch == -0.0:
            assert trace.epoch != mount.epoch_token(0.0) and trace.epoch != 0

    page = _page()
    try:
        page._open_analysis_mount("peak")
        dialog = page._peak_dialog
        trace = _trace(); anchor = SimpleNamespace(frame=_frame())
        monkeypatch.setattr(page, "_capture_analysis_trace",
                            lambda kind: (trace, anchor))
        monkeypatch.setattr(mount, "display_anchor_matches",
                            lambda *args: True)
        dialog.set_vnext_trace(trace)
        dialog.auto_check.setChecked(False)
        dialog.npeaks_spin.setValue(2)
        dialog.range_lo.setText("1.5"); dialog.range_hi.setText("3")
        dialog._fields_to_region()
        dialog.adv_centers.setText("")
        plans = []
        identity = OperationIdentity(71)
        monkeypatch.setattr(page._analysis_slot, "begin_peak_fit",
            lambda plan, stamp: plans.append(plan) or identity)
        dialog.fit_btn.click()
        plan = plans[-1]
        assert plan.trace is trace and plan.manual_centers == ()
        assert plan.fit_bounds == (1.5, 3.0)
        assert plan.selection_mode == "count" and plan.n_peaks == 2
        result = DisplayedPeakFitResult(
            AnalysisDisposition.COMPLETED, "OK", diagnostics=("peak-detail",),
            trace_fingerprint=trace.trace_fingerprint,
            trace_receipt=trace.receipt, plan_fingerprint="headless-plan",
            policy_fingerprint="headless-policy", fit_success=True,
            message="fit complete", label=trace.label,
            axis_unit=trace.axis_unit,
            parameter_names=("center", "sigma"),
            parameter_values=(2.1, 0.2), parameter_stderr=(0.01, None),
            fit=np.array([8.0, 9.0]), background=np.array([1.0, 1.0]),
            residual=np.array([0.1, -0.1]), marker_positions=np.array([2.1]),
            storage_bytes=128, result_fingerprint="peak-result")
        assert page._consume_analysis_update(_returned(identity, result))
        assert page._peak_result is result
        projected_x = [item.getData()[0] for item in dialog.plot.listDataItems()]
        assert any(np.array_equal(values, np.array([2.0, 3.0]))
                   for values in projected_x)
        assert dialog.resid_plot.listDataItems()
        assert dialog.table.rowCount() == 2
        assert "peak-detail" in dialog.status.text()
        assert "headless-plan" in dialog.status.toolTip()
    finally:
        _close_page(page, qapp)


def test_p37b_phase_captures_paths_only_and_preserves_q_refusal_cif_receipts(
        monkeypatch, qapp) -> None:
    import xdart.gui.tabs.scattering.analysis_mount as mount

    for wavelength_m, expected in ((1.234e-10, 1.234), (1.0e-10, 1.0)):
        calls = []
        controller = SimpleNamespace(
            project_request=lambda frame, require_complete=True: calls.append(
                (frame, require_complete)) or object(),
            resolve_projection=lambda request, value=wavelength_m:
                SimpleNamespace(wavelength_m=value),
        )
        assert mount.phase_wavelength_angstrom(controller, _frame()) == expected
        assert calls == [(_frame(), False)]
    controller.resolve_projection = lambda request: SimpleNamespace(wavelength_m=None)
    assert mount.phase_wavelength_angstrom(controller, _frame()) is None

    page = _page()
    try:
        page._open_analysis_mount("phase")
        dialog = page._phase_dialog
        trace = _trace(); anchor = SimpleNamespace(frame=_frame())
        monkeypatch.setattr(page, "_capture_analysis_trace",
                            lambda kind: (trace, anchor))
        monkeypatch.setattr(mount, "display_anchor_matches",
                            lambda *args: True)
        monkeypatch.setattr(mount, "phase_wavelength_angstrom",
                            lambda controller, frame: 1.0)
        too_long = "/tmp/" + "x" * 257 + ".cif"
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
            lambda *args, **kwargs: (too_long, "CIF"))
        dialog._add_cif()
        assert not dialog._phases
        assert dialog.status.text() == "PHASE_NAME_LIMIT_EXCEEDED"
        choices = iter(("/tmp/alpha.cif", "/tmp/beta.cif", "/tmp/alpha.cif"))
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
            lambda *args, **kwargs: (next(choices), "CIF"))
        dialog._add_cif(); dialog._add_cif(); dialog._add_cif()
        assert dialog._phases == [
            ("/tmp/alpha.cif", "alpha"), ("/tmp/beta.cif", "beta")]
        assert "Duplicate" in dialog.status.text()
        dialog.set_vnext_trace(trace)
        plans = []
        identity = OperationIdentity(81)
        monkeypatch.setattr(page._analysis_slot, "begin_phase_fit",
            lambda plan, stamp: plans.append(plan) or identity)
        dialog.fit_btn.click()
        plan = plans[-1]
        assert plan.trace is trace
        assert plan.cif_paths == (Path("/tmp/alpha.cif"), Path("/tmp/beta.cif"))
        assert plan.phase_names == ("alpha", "beta")
        assert plan.wavelength_angstrom == 1.0
        states = (1, 2, 3, 4, 5, 6)
        receipts = tuple(CifAssetReceipt(
            Path(f"/tmp/{name}.cif"), Path(f"/resolved/{name}.cif"), name,
            100 + index, f"sha-{name}", None, states, states,
            f"receipt-{name}") for index, name in enumerate(("alpha", "beta")))
        result = DisplayedPhaseFitResult(
            AnalysisDisposition.COMPLETED, "OK", diagnostics=("phase-detail",),
            trace_fingerprint=trace.trace_fingerprint,
            trace_receipt=trace.receipt, plan_fingerprint="phase-plan",
            policy_fingerprint="phase-policy", cif_receipts=receipts,
            wavelength_angstrom=1.0, fit_success=True,
            message="phase fit complete", label=trace.label,
            axis_unit=trace.axis_unit, parameter_names=("scale",),
            parameter_values=(3.0,), parameter_stderr=(0.2,),
            phase_fractions=(("alpha", 0.6), ("beta", 0.4)),
            lattice_parameters=(("alpha", (("a", 4.1),)),
                                ("beta", (("a", 5.2),))),
            phase_components=(np.array([3.0, 4.0, 5.0]),
                              np.array([2.0, 2.0, 2.0])),
            fit=np.array([5.0, 6.0, 7.0]),
            background=np.array([1.0, 1.0, 1.0]),
            residual=np.array([0.1, 0.0, -0.1]),
            marker_positions=np.array([1.5, 2.5]), storage_bytes=256,
            result_fingerprint="phase-result")
        assert page._consume_analysis_update(_returned(identity, result))
        assert page._phase_result is result
        assert len(dialog.plot.listDataItems()) >= 5
        assert dialog.resid_plot.listDataItems()
        assert dialog.table.rowCount() == 3
        assert "sha-alpha" in dialog.cif_list.item(0).toolTip()
        assert "phase-detail" in dialog.status.text()
        assert "phase-plan" in dialog.status.toolTip()

        more = iter(f"/tmp/{name}.cif" for name in "cdefgh")
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
            lambda *args, **kwargs: (next(more), "CIF"))
        for _ in range(6): dialog._add_cif()
        assert len(dialog._phases) == 8
        dialog._add_cif()
        assert dialog.status.text() == "PHASE_COUNT_LIMIT_EXCEEDED"
    finally:
        _close_page(page, qapp)


def test_p37b_context_fingerprints_and_dialog_generation_jointly_gate_adoption(
        monkeypatch, qapp, tmp_path) -> None:
    import xdart.gui.tabs.scattering.analysis_mount as mount

    frame = _frame(); gate = SimpleNamespace(cancelled=False)
    context = SimpleNamespace(loaded=True, invalidated=False, released=False,
                              commit_gate=gate)
    selection = SimpleNamespace(context_token="ctx", display_generation=4,
                                names=lambda candidate: candidate is context)
    controller = SimpleNamespace(
        selection=selection, navigation=SimpleNamespace(current=frame),
        owns_frame=lambda candidate: candidate is frame,
        projectable_contexts=(context,), browse_context=context,
        browse_pending=False)
    anchor_page = SimpleNamespace(_context_controller=controller)
    real_anchor = mount.capture_display_anchor(anchor_page, "trace-fp", 3)
    assert real_anchor is not None
    assert mount.display_anchor_matches(anchor_page, real_anchor, "trace-fp", 3)
    context.invalidated = True
    assert not mount.display_anchor_matches(anchor_page, real_anchor, "trace-fp", 3)
    context.invalidated = False; gate.cancelled = True
    assert not mount.display_anchor_matches(anchor_page, real_anchor, "trace-fp", 3)
    gate.cancelled = False; selection.display_generation = 5
    assert not mount.display_anchor_matches(anchor_page, real_anchor, "trace-fp", 3)

    artifact = tmp_path / "input.nxs"
    artifact.write_bytes(b"stable metadata artifact")
    table = _table(
        artifact, primary_post_state=_file_revision(artifact),
    )
    receipt = table.receipt; trace = _trace()
    assert receipt is not None
    signal = RoiSignal(RoiSpec.full_frame("roi"), name="roi")
    plans = {
        "metadata": MetadataTablePlan(artifact),
        "scan_plot": ScanPlotPlan("motor", ("signal",), None),
        "roi_preview": RoiPreviewPlan.from_table(table, label=1),
        "roi_scan": RoiScanPlan.from_table(
            table, signals=(signal,), mask_saturation=True),
        "peak": DisplayedPeakFitPlan(trace, fit_bounds=(1.0, 3.0)),
        "phase": DisplayedPhaseFitPlan(
            trace, (Path("/tmp/a.cif"),), (None,), ("a",), 1.0),
    }
    results = {
        "metadata": table,
        "scan_plot": ScanPlotResult(
            AnalysisDisposition.COMPLETED, "OK",
            table_fingerprint=table.table_fingerprint, x_name="motor",
            x=np.arange(3.0), trace_names=("signal",),
            trace_origins=("metadata",), original_identities=("signal",),
            traces=(np.arange(3.0),), storage_bytes=64),
        "roi_preview": RoiPreviewResult(
            AnalysisDisposition.COMPLETED, "OK", receipt=receipt,
            table_fingerprint=table.table_fingerprint, labels=table.labels,
            label=1, image=np.ones((2, 2)), result_fingerprint="preview"),
        "roi_scan": RoiScanResult(
            AnalysisDisposition.COMPLETED, "OK", receipt=receipt,
            table_fingerprint=table.table_fingerprint,
            requested_labels=table.labels, completed_labels=table.labels,
            signal_names=("roi",), signal_values=(np.ones(3),),
            valid_counts=(np.ones(3, dtype=int),), result_fingerprint="roi"),
        "peak": DisplayedPeakFitResult(
            AnalysisDisposition.COMPLETED, "OK",
            trace_fingerprint=trace.trace_fingerprint,
            trace_receipt=trace.receipt, plan_fingerprint="opaque-peak"),
        "phase": DisplayedPhaseFitResult(
            AnalysisDisposition.COMPLETED, "OK",
            trace_fingerprint=trace.trace_fingerprint,
            trace_receipt=trace.receipt, plan_fingerprint="opaque-phase"),
    }
    facts = {
        "metadata": mount.analysis_request_facts(plans["metadata"]),
        "scan_plot": mount.analysis_request_facts(
            plans["scan_plot"], table=table, render=((), False)),
        "roi_preview": mount.analysis_request_facts(plans["roi_preview"]),
        "roi_scan": mount.analysis_request_facts(
            plans["roi_scan"], picker=("preview", 7)),
        "peak": mount.analysis_request_facts(plans["peak"]),
        "phase": mount.analysis_request_facts(plans["phase"]),
    }
    targets = {"metadata": "metadata", "scan_plot": "scan_roi",
               "roi_preview": "scan_roi", "roi_scan": "scan_roi",
               "peak": "peak", "phase": "phase"}
    extra = {"scan_plot": {"table": table, "roi": None}}
    anchor = SimpleNamespace(frame=_frame())
    page = _page()
    try:
        page._open_analysis_mount("metadata")
        starts = []
        for name in ("begin_metadata", "begin_scan_plot", "begin_roi_preview",
                     "begin_roi_scan", "begin_peak_fit", "begin_phase_fit"):
            monkeypatch.setattr(page._analysis_slot, name,
                lambda *args, owner=name: starts.append(owner) or OperationIdentity(999))
        monkeypatch.setattr(mount, "display_anchor_matches", lambda *args: True)

        for kind, request in facts.items():
            assert request is not None
            for index in range(len(request[1])):
                changed_row = (*request[1][:index], ("changed", index),
                               *request[1][index + 1:])
                changed = (request[0], changed_row, request[2])
                with monkeypatch.context() as patch:
                    patch.setattr(page, "_current_analysis_request",
                                  lambda _kind, _target, value=changed: value)
                    if kind == "metadata":
                        assert page._submit_metadata(
                            plans[kind],
                            target=targets[kind],
                            request=request,
                        ) is None
                    else:
                        assert page._begin_analysis(
                            kind, plans[kind],
                            page._analysis_generation_for(targets[kind]),
                            target=targets[kind], request=request,
                            anchor=anchor,
                            **extra.get(kind, {}),
                        ) is None
        assert not starts

        for kind, request in facts.items():
            payload = results[kind]
            assert mount.analysis_result_matches(payload, request)
            assert not mount.analysis_result_matches(object(), request)
            adopted, diagnostic = mount.terminal_adoption(
                _returned(OperationIdentity(90), payload), current=True)
            assert adopted is payload and diagnostic == ""
        mismatches = {
            "metadata": (
                replace(table, table_fingerprint=""),
                replace(table, receipt=_receipt("/tmp/other.nxs")),
            ),
            "scan_plot": (
                replace(results["scan_plot"], table_fingerprint="other"),
                replace(results["scan_plot"], roi_fingerprint="other"),
                replace(results["scan_plot"], x_name="other"),
                replace(results["scan_plot"], original_identities=("other",)),
                replace(results["scan_plot"], normalization="signal"),
            ),
            "roi_preview": (
                replace(results["roi_preview"], receipt=_receipt("/tmp/other.nxs")),
                replace(results["roi_preview"], table_fingerprint="other"),
                replace(results["roi_preview"], labels=(1, 2)),
                replace(results["roi_preview"], label=2),
            ),
            "roi_scan": (
                replace(results["roi_scan"], receipt=_receipt("/tmp/other.nxs")),
                replace(results["roi_scan"], table_fingerprint="other"),
                replace(results["roi_scan"], requested_labels=(1, 2)),
                replace(results["roi_scan"], signal_names=("other",)),
            ),
            "peak": (
                replace(results["peak"], trace_fingerprint="other"),
                replace(results["peak"], trace_receipt=None),
            ),
            "phase": (
                replace(results["phase"], trace_fingerprint="other"),
                replace(results["phase"], trace_receipt=None),
            ),
        }
        for kind, payloads in mismatches.items():
            for payload in payloads:
                assert not mount.analysis_result_matches(payload, facts[kind])
        directory_facts = mount.analysis_request_facts(MetadataTablePlan(
            "/tmp", selection="directory", kind=SourceKind.PROCESSED_NEXUS))
        assert not mount.analysis_result_matches(table, directory_facts)
        assert mount.analysis_result_matches(results["peak"], facts["peak"])
        assert results["peak"].plan_fingerprint == "opaque-peak"

        exact = OperationIdentity(91)
        exact_process = _start_metadata_process(
            page, exact, plans["metadata"], facts["metadata"],
        )
        assert page._consume_analysis_update(
            _returned(OperationIdentity(92), table)
        ) is WorkspaceRefreshEffect.NONE
        assert page._metadata_operations.active is exact_process
        requalification_identity = OperationIdentity(999)
        with monkeypatch.context() as patch:
            patch.setattr(page, "_current_analysis_request",
                          lambda kind, target: facts["metadata"])
            patch.setattr(
                page._analysis_slot, "begin_metadata_requalification",
                lambda plan, stamp: requalification_identity,
            )
            assert page._consume_analysis_update(_returned(exact, table))
            assert page._metadata_result is None
            assert (
                page._metadata_operations.active_identity
                is requalification_identity
            )
            assert page._consume_analysis_update(_returned(
                requalification_identity,
                MetadataTableRequalificationResult(
                    AnalysisDisposition.COMPLETED, "OK", table.receipt,
                    table.table_fingerprint,
                ),
            ))
        assert page._metadata_result is table

        mismatch_identity = OperationIdentity(93)
        _start_metadata_process(
            page,
            mismatch_identity,
            plans["metadata"],
            facts["metadata"],
        )
        mismatched_table = replace(
            table, receipt=_receipt("/tmp/mismatched.nxs"))
        with monkeypatch.context() as patch:
            patch.setattr(page, "_current_analysis_request",
                          lambda kind, target: facts["metadata"])
            assert page._consume_analysis_update(
                _returned(mismatch_identity, mismatched_table))
        assert page._metadata_result is table
        assert page._notice_text == "P3_7_ANALYSIS_RESULT_IDENTITY_MISMATCH"

        serial = 100
        for kind, request in facts.items():
            for index in range(len(request[1])):
                serial += 1; identity = OperationIdentity(serial)
                changed_row = (*request[1][:index], ("post-start", index),
                               *request[1][index + 1:])
                changed = (request[0], changed_row, request[2])
                target = targets[kind]
                if kind == "metadata":
                    _start_metadata_process(
                        page, identity, plans[kind], request, target=target,
                    )
                else:
                    page._analysis_identity = identity
                    page._analysis_kind = kind
                    page._analysis_target = target
                    page._analysis_generation = (
                        page._analysis_generation_for(target)
                    )
                    page._analysis_request = request
                    page._analysis_anchor = anchor
                    page._analysis_fingerprint = trace.trace_fingerprint
                with monkeypatch.context() as patch:
                    patch.setattr(page, "_current_analysis_request",
                                  lambda _kind, _target, value=changed: value)
                    assert page._consume_analysis_update(
                        _returned(identity, results[kind]))
                assert page._notice_text == "P3_7_ANALYSIS_STALE"
    finally:
        _close_page(page, qapp)


def test_p37b_typed_terminal_projection_never_adopts_refused_cancelled_failed() -> None:
    from xdart.gui.tabs.scattering.analysis_mount import terminal_adoption

    identity = OperationIdentity(1)
    for disposition, status in (
        (AnalysisDisposition.REFUSED, OperationTerminalStatus.RETURNED),
        (AnalysisDisposition.CANCELLED, OperationTerminalStatus.CANCELLED),
    ):
        payload = MetadataTableResult(disposition, disposition.value.upper())
        update = OperationUpdate(identity, terminal=OperationTerminal(
            identity, status, payload=payload))
        adopted, diagnostic = terminal_adoption(update, current=True)
        assert adopted is None and diagnostic == disposition.value.upper()
    failed = OperationUpdate(identity, terminal=OperationTerminal(
        identity, OperationTerminalStatus.FAILED, "boom"))
    assert terminal_adoption(failed, current=True) == (None, "boom")
    completed = MetadataTableResult(AnalysisDisposition.COMPLETED, "OK")
    stale = OperationUpdate(identity, terminal=OperationTerminal(
        identity, OperationTerminalStatus.RETURNED, payload=completed), stale=True)
    assert terminal_adoption(stale, current=True) == (
        None, "P3_7_ANALYSIS_STALE")


def test_p37b_dialog_and_app_close_cancel_without_abandoning_source_cleanup(
        qapp) -> None:
    page = _page(); entered = Event(); release = Event()
    try:
        for target in ("metadata", "scan_roi", "peak", "phase"):
            page._open_analysis_mount(target)
        dialogs = (page._metadata_dialog, page._scan_roi_dialog,
                   page._peak_dialog, page._phase_dialog)
        assert all(dialog is not None and dialog.testAttribute(
            QtCore.Qt.WidgetAttribute.WA_DeleteOnClose) for dialog in dialogs)
        destroyed = [False] * 4
        for index, dialog in enumerate(dialogs):
            dialog.destroyed.connect(
                lambda *_args, row=index: destroyed.__setitem__(row, True))

        @dataclass(frozen=True)
        class Job: pass
        def body(job, identity, cancelled, publish):
            entered.set(); release.wait(2)
            assert cancelled.is_set()
            return OperationTerminal(identity, OperationTerminalStatus.CANCELLED,
                payload=MetadataTableResult(
                    AnalysisDisposition.CANCELLED, "CANCELLED"))
        identity = page._analysis_slot._begin(
            Job(), OperationContextStamp(0), body)
        assert type(identity) is OperationIdentity and entered.wait(2)
        plan = MetadataTablePlan("/tmp/cancel.nexus")
        from xdart.gui.tabs.scattering.analysis_mount import (
            analysis_request_facts,
        )
        token = page._metadata_operations.dialog_identity("scan_roi")
        captured = page._metadata_operations.capture(
            plan,
            analysis_request_facts(plan),
            target="scan_roi",
            dialog=token,
        )
        assert page._metadata_operations.start(
            captured, identity, OperationContextStamp(0),
        ) is not None
        dialogs[1].close()
        assert page._scan_roi_dialog is None and page._analysis_slot.owned
        release.set(); update = _finish(page._analysis_slot, identity)
        assert update.terminal.status is OperationTerminalStatus.CANCELLED
        assert page._consume_analysis_update(
            update
        ) is WorkspaceRefreshEffect.CONTROLS
        assert page._metadata_operations.active is None

        for dialog in (dialogs[0], dialogs[2], dialogs[3]): dialog.close()
        QtCore.QCoreApplication.sendPostedEvents(
            None, QtCore.QEvent.Type.DeferredDelete)
        qapp.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(
            None, QtCore.QEvent.Type.DeferredDelete)
        assert destroyed == [True, True, True, True]
        assert (page._metadata_dialog, page._scan_roi_dialog,
                page._peak_dialog, page._phase_dialog) == (None, None, None, None)
    finally:
        release.set(); _close_page(page, qapp)


def test_metadata_page_close_retries_exact_cleanup_to_closed(qapp) -> None:
    page = _page()
    entered = Event()
    release = Event()

    @dataclass(frozen=True)
    class Job:
        pass

    def body(job, identity, cancelled, publish):
        entered.set()
        assert release.wait(2)
        assert cancelled.is_set()
        return OperationTerminal(
            identity,
            OperationTerminalStatus.CANCELLED,
            payload=MetadataTableResult(
                AnalysisDisposition.CANCELLED,
                "CANCELLED",
            ),
        )

    try:
        page._open_analysis_mount("metadata")
        identity = page._analysis_slot._begin(
            Job(), OperationContextStamp(0), body,
        )
        assert type(identity) is OperationIdentity and entered.wait(2)
        plan = MetadataTablePlan("/tmp/close-retry.nexus")
        from xdart.gui.tabs.scattering.analysis_mount import (
            analysis_request_facts,
        )
        token = page._metadata_operations.dialog_identity("metadata")
        captured = page._metadata_operations.capture(
            plan,
            analysis_request_facts(plan),
            target="metadata",
            dialog=token,
        )
        assert page._metadata_operations.start(
            captured, identity, OperationContextStamp(0),
        ) is not None

        first = page.close_workspace()
        assert first.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert page._metadata_operations.lifecycle is MetadataLifecycle.CLOSING
        assert not page._closed

        release.set()
        worker = page._analysis_slot._worker
        assert worker is not None
        worker.join(2)
        assert not worker.is_alive()
        second = page.close_workspace()
        assert second.cleanup_status is CleanupStatus.CLEANED
        assert page._metadata_operations.lifecycle is MetadataLifecycle.CLOSED
        assert page._closed
    finally:
        release.set()
        if not page._closed:
            page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_p37b_close_reopen_reuses_only_self_contained_metadata_and_discharges_transient_results(
        monkeypatch, qapp) -> None:
    from xdart.gui.tabs.scattering.analysis_mount import analysis_request_facts

    page = _page(); table = _table(fingerprint="reopen-table")
    facts = analysis_request_facts(
        MetadataTablePlan(table.receipt.source_spec)
    )
    identities = iter((OperationIdentity(170), OperationIdentity(171)))
    monkeypatch.setattr(
        page,
        "_current_analysis_request",
        lambda kind, target: facts
        if kind in {"metadata", "metadata_requalification"}
        else None,
    )
    monkeypatch.setattr(
        page._analysis_slot,
        "begin_metadata_requalification",
        lambda plan, stamp: next(identities),
    )

    def finish_requalification():
        identity = page._metadata_operations.active_identity
        assert identity is not None
        assert page._consume_analysis_update(_returned(
            identity,
            MetadataTableRequalificationResult(
                AnalysisDisposition.COMPLETED,
                "OK",
                table.receipt,
                table.table_fingerprint,
            ),
        ))

    try:
        page._metadata_result = table; page._open_analysis_mount("metadata")
        metadata = page._metadata_dialog
        assert metadata.table.rowCount() == 0
        finish_requalification()
        assert metadata.table.rowCount() == len(table.labels)
        metadata.close(); qapp.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        assert page._metadata_dialog is None and page._metadata_result is table; page._open_analysis_mount("metadata")
        assert page._metadata_dialog.table.rowCount() == 0
        finish_requalification()
        assert (page._metadata_dialog is not metadata
                and page._metadata_dialog.table.rowCount() == len(table.labels))

        page._open_analysis_mount("scan_roi"); scan_dialog = page._scan_roi_dialog
        assert scan_dialog._vnext_table_result is table
        scan = ScanPlotResult(
            AnalysisDisposition.COMPLETED, "OK",
            table_fingerprint=table.table_fingerprint, x_name="motor",
            x=np.arange(3.0), trace_names=("signal",),
            original_identities=("signal",), traces=(np.arange(3.0),),
            storage_bytes=64)
        page._scan_roi_result = scan
        scan_dialog.set_vnext_scan_result(
            scan, (table.table_fingerprint, None, "motor", ("signal",), None))
        page._roi_preview_binding = (object(),)
        scan_dialog.close(); qapp.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        assert page._scan_roi_result is page._roi_preview_binding is None
        assert page._metadata_result is table
        page._open_analysis_mount("scan_roi")
        assert page._scan_roi_dialog._vnext_table_result is table
        assert (page._scan_roi_dialog._vnext_scan_result is
                page._scan_roi_dialog._vnext_roi_result is None)

        for target, field in (("peak", "_peak_result"),
                              ("phase", "_phase_result")):
            page._open_analysis_mount(target)
            dialog = getattr(page, f"_{target}_dialog")
            setattr(page, field, SimpleNamespace(storage_bytes=32))
            dialog.close(); qapp.processEvents()
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
            assert getattr(page, field) is None; page._open_analysis_mount(target)
            assert (getattr(page, f"_{target}_dialog") is not dialog
                    and getattr(page, field) is None)
    finally:
        _close_page(page, qapp)


def test_p37b_shared_scan_roi_replacement_retires_prior_payload_graph_only_after_admission(
        monkeypatch, qapp) -> None:
    from xdart.gui.tabs.scattering.analysis_mount import analysis_request_facts

    page = _page(); serial = 200
    try:
        page._open_analysis_mount("scan_roi"); dialog = page._scan_roi_dialog
        table = replace(_table(fingerprint="retained-table"),
                        storage_bytes=64 << 20)
        page._metadata_result = table; dialog.set_vnext_metadata(table)
        dialog._vnext_rendering = True
        try: _check_item(dialog.r_list, "signal")
        finally: dialog._vnext_rendering = False

        current_request = [None]

        def begin_requalification(_plan, _stamp):
            nonlocal serial
            serial += 1
            return OperationIdentity(serial)

        monkeypatch.setattr(
            page._analysis_slot,
            "begin_metadata_requalification",
            begin_requalification,
        )

        def consume(kind, payload, request, *, target="scan_roi"):
            nonlocal serial
            current_request[0] = request
            serial += 1; identity = OperationIdentity(serial)
            if kind == "metadata":
                plan = MetadataTablePlan(
                    payload.receipt.source_spec
                    if payload.receipt is not None
                    else "/tmp/refused.nexus"
                )
                _start_metadata_process(
                    page, identity, plan, request, target=target,
                )
                effect = page._consume_analysis_update(
                    _returned(identity, payload)
                )
                active = page._metadata_operations.active
                if (
                    type(payload) is MetadataTableResult
                    and payload.disposition is AnalysisDisposition.COMPLETED
                    and active is not None
                    and active.request.candidate is payload
                ):
                    effect = page._consume_analysis_update(_returned(
                        active.identity,
                        MetadataTableRequalificationResult(
                            AnalysisDisposition.COMPLETED,
                            "OK",
                            payload.receipt,
                            payload.table_fingerprint,
                        ),
                    ))
                return effect
            page._analysis_identity = identity; page._analysis_kind = kind
            page._analysis_target = target
            page._analysis_generation = page._analysis_generation_for(target)
            page._analysis_request = request
            return page._consume_analysis_update(_returned(identity, payload))

        monkeypatch.setattr(
            page,
            "_current_analysis_request",
            lambda _kind, _target: current_request[0],
        )
        scan_request = ("scan_plot", (), (table.table_fingerprint, None,
                        "motor", ("signal",), None))
        scan = ScanPlotResult(
            AnalysisDisposition.COMPLETED, "OK",
            table_fingerprint=table.table_fingerprint, x_name="motor",
            x=np.arange(3.0), trace_names=("signal",),
            original_identities=("signal",), traces=(np.arange(3.0),),
            storage_bytes=64 << 20)
        assert consume("scan_plot", scan, scan_request)
        old_items = tuple(dialog.plot.listDataItems()); old_right = tuple(
            dialog.right_vb.addedItems); old_request = dialog._vnext_scan_request

        roi_request = ("roi_scan", (), (table.receipt,
                       table.table_fingerprint, table.labels, ("roi",)))
        over_roi = RoiScanResult(
            AnalysisDisposition.COMPLETED, "OK", receipt=table.receipt,
            table_fingerprint=table.table_fingerprint,
            requested_labels=table.labels, completed_labels=table.labels,
            signal_names=("roi",), signal_values=(np.ones(3),),
            valid_counts=(np.ones(3, dtype=int),), storage_bytes=(64 << 20) + 1)
        assert consume("roi_scan", over_roi, roi_request)
        assert page._scan_roi_result is dialog._vnext_scan_result is scan
        assert dialog._vnext_scan_request is old_request
        assert (tuple(dialog.plot.listDataItems()), tuple(dialog.right_vb.addedItems)) == (
            old_items, old_right)

        roi = replace(over_roi, storage_bytes=64 << 20)
        assert consume("roi_scan", roi, roi_request)
        assert page._scan_roi_result is dialog._vnext_roi_result is roi
        assert dialog._vnext_scan_result is dialog._vnext_scan_request is None
        assert not dialog.plot.listDataItems() and not dialog.right_vb.addedItems
        retained = (page._metadata_result, page._scan_roi_result,
                    dialog._vnext_table_result, dialog._vnext_scan_result,
                    dialog._vnext_roi_result)
        unique = {id(value): value for value in retained if value is not None}
        assert sum(value.storage_bytes for value in unique.values()) == 128 << 20

        assert consume("scan_plot", scan, scan_request)
        assert (page._scan_roi_result is dialog._vnext_scan_result is scan
                and dialog._vnext_roi_result is None)
        preview_request = ("roi_preview", (), (table.receipt,
                           table.table_fingerprint, table.labels, table.labels[0]))
        preview = RoiPreviewResult(
            AnalysisDisposition.COMPLETED, "OK", receipt=table.receipt,
            table_fingerprint=table.table_fingerprint, labels=table.labels,
            label=table.labels[0], image=np.ones((3, 3)),
            result_fingerprint="retained-preview", storage_bytes=64 << 20)
        assert consume("roi_preview", preview, preview_request)
        picker = dialog._roi_dialog; binding = page._roi_preview_binding; destroyed = []
        picker.destroyed.connect(lambda *_args: destroyed.append(True))

        page._peak_result = SimpleNamespace(storage_bytes=64 << 20)
        over_table = replace(_table(fingerprint="over-table"),
                             storage_bytes=(64 << 20) + 1)
        over_facts = analysis_request_facts(
            MetadataTablePlan(over_table.receipt.source_spec))
        assert consume("metadata", over_table, over_facts)
        assert page._scan_roi_result is preview and page._roi_preview_binding is binding
        assert dialog._roi_dialog is picker and not destroyed
        page._peak_result = None

        replacement = replace(_table(fingerprint="replacement-table"),
                              storage_bytes=64 << 20)
        replacement_facts = analysis_request_facts(
            MetadataTablePlan(replacement.receipt.source_spec))
        assert consume("metadata", replacement, replacement_facts)
        assert page._metadata_result is dialog._vnext_table_result is replacement
        assert page._scan_roi_result is page._roi_preview_binding is None
        assert dialog._roi_dialog is None
        assert not dialog.plot.listDataItems() and not dialog.right_vb.addedItems
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        qapp.processEvents()
        assert destroyed == [True]

        page._scan_roi_result = scan
        dialog.set_vnext_scan_result(scan, scan_request[2])
        old_metadata = page._metadata_result
        refused = MetadataTableResult(
            AnalysisDisposition.REFUSED, "SOURCE_SELECTION_REQUIRED",
            candidates=(CandidateProjection(
                SourceSpec("/tmp/candidate.nxs", SourceKind.PROCESSED_NEXUS),
                "candidate", 1),))
        assert consume("metadata", refused, ("metadata", (), ()),
                       target="scan_roi")
        assert page._metadata_result is old_metadata
        assert page._scan_roi_result is page._roi_preview_binding is None
        assert dialog._vnext_table_result is dialog._vnext_scan_result is None
        assert not dialog.plot.listDataItems() and not dialog.right_vb.addedItems

        dialog.set_vnext_metadata(replacement)
        page._scan_roi_result = scan
        dialog.set_vnext_scan_result(scan, scan_request[2])
        current_request[0] = replacement_facts
        page._open_analysis_mount("metadata")
        retained_requalification = page._metadata_operations.active
        assert retained_requalification is not None
        assert page._consume_analysis_update(_returned(
            retained_requalification.identity,
            MetadataTableRequalificationResult(
                AnalysisDisposition.COMPLETED,
                "OK",
                replacement.receipt,
                replacement.table_fingerprint,
            ),
        ))
        standalone = replace(_table(fingerprint="standalone-table"),
                             storage_bytes=64 << 20)
        standalone_facts = analysis_request_facts(
            MetadataTablePlan(standalone.receipt.source_spec))
        assert consume("metadata", standalone, standalone_facts,
                       target="metadata")
        assert page._metadata_result is standalone
        assert page._scan_roi_result is None
        assert dialog._vnext_table_result is None
        assert not dialog.plot.listDataItems() and not dialog.right_vb.addedItems
    finally:
        _close_page(page, qapp)


def test_p37b_gui_retention_cap_refuses_new_adoption_without_silent_eviction() -> None:
    from xdart.gui.tabs.scattering.analysis_mount import retention_admission

    @dataclass(frozen=True)
    class Result:
        storage_bytes: int
    current = {"metadata": Result(40 << 20), "scan_roi": Result(40 << 20),
               "peak": Result(40 << 20), "phase": None}
    assert retention_admission(current, "phase", Result(8 << 20)) == (True, "")
    before = dict(current)
    assert retention_admission(current, "phase", Result((8 << 20) + 1)) == (
        False, "P3_7_GUI_RETENTION_LIMIT")
    assert current == before


def test_p37b_batch_live_texture_export_writer_and_private_average_are_absent(
        qapp) -> None:
    from xdart.gui.tabs.scattering.analysis_mount import HELD_REASONS
    from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    from xdart.gui.analysis.scan_plot_dialog import ScanPlotDialog

    assert HELD_REASONS == {
        "peak_live": "P3_7_LIVE_FITTING_UNAVAILABLE",
        "peak_batch": "P3_7_BATCH_DISPLAY_PROJECTION_UNAVAILABLE",
        "phase_batch": "P3_7_BATCH_DISPLAY_PROJECTION_UNAVAILABLE",
        "phase_texture": "P3_7_PHASE_TEXTURE_UNAVAILABLE",
        "export": "P3_7_EXPORT_UNAVAILABLE",
    }
    text = Path(importlib.import_module(
        "xdart.gui.tabs.scattering.analysis_mount").__file__).read_text()
    assert not {"h5py", "NexusRecordWriter", "average_closed_v1"} & set(text.split())
    submit = lambda *_args: None
    dialogs = (ScanPlotDialog(vnext_submit=submit),
               PeakFitDialog(vnext_submit=submit),
               PhaseFitDialog(vnext_submit=submit))
    try:
        scan, peak, phase = dialogs
        assert not scan.save_btn.isEnabled()
        assert scan.save_btn.toolTip() == HELD_REASONS["export"]
        assert not peak.live_check.isEnabled() and not peak.live_check.isChecked()
        assert peak.live_check.toolTip() == HELD_REASONS["peak_live"]
        assert not peak.batch_btn.isEnabled()
        assert peak.batch_btn.toolTip() == HELD_REASONS["peak_batch"]
        assert not phase.texture_combo.isEnabled()
        assert phase.texture_combo.toolTip() == HELD_REASONS["phase_texture"]
        assert not phase.batch_btn.isEnabled()
        assert phase.batch_btn.toolTip() == HELD_REASONS["phase_batch"]
    finally:
        for dialog in dialogs: dialog.close()
        qapp.processEvents()


def test_p37b_cold_import_and_ordinary_page_paths_are_analysis_dormant() -> None:
    path = Path(__file__).parents[3] / "src/xdart/gui/tabs/scattering/page.py"
    tree = ast.parse(path.read_text())
    allocations = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and getattr(node.func, "id", None) == "OperationSlot"]
    assert len(allocations) == 1
    text = path.read_text()
    assert "xrd_tools.analysis.display_fit_operations import" not in text
    assert "QThread(" not in text and "analysis_timer" not in text


def test_p37b_legacy_dialog_defaults_remain_reachable_only_for_canonical_callers(qapp) -> None:
    from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    from xdart.gui.analysis.scan_plot_dialog import ScanPlotDialog

    dialogs = [ScanPlotDialog(), PeakFitDialog(), PhaseFitDialog()]
    try:
        assert all(not getattr(dialog, "_vnext", False) for dialog in dialogs)
        assert dialogs[0].save_btn.isEnabled() is False
        assert dialogs[1].batch_btn.isEnabled()
        assert dialogs[2].texture_combo.isEnabled()
    finally:
        for dialog in dialogs: dialog.close()
        qapp.processEvents()
