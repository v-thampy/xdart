"""Thin Qt capture, identity-check, adoption, and detached-render helpers."""
from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
import struct
from typing import Mapping
from pyqtgraph.Qt import QtWidgets
from .operation_values import OperationTerminalStatus, OperationUpdate
from .shell_values import ShellCommandKind
MOUNT_TARGETS = {
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
HELD_REASONS = {
    "peak_live": "P3_7_LIVE_FITTING_UNAVAILABLE",
    "peak_batch": "P3_7_BATCH_DISPLAY_PROJECTION_UNAVAILABLE",
    "phase_batch": "P3_7_BATCH_DISPLAY_PROJECTION_UNAVAILABLE",
    "phase_texture": "P3_7_PHASE_TEXTURE_UNAVAILABLE",
    "export": "P3_7_EXPORT_UNAVAILABLE",
}
RETENTION_LIMIT = 128 << 20
def mount_target(kind: object, value: object) -> str | None:
    key = (kind, "" if kind is ShellCommandKind.SHOW_METADATA else str(value))
    return MOUNT_TARGETS.get(key)
def analysis_start_allowed(page: object) -> bool:
    lifecycle = getattr(page, "_lifecycle", None)
    controller = getattr(page, "_context_controller", None)
    slot = getattr(page, "_analysis_slot", None)
    experiment_busy = getattr(page, "_experiment_operation_busy", None)
    phase = getattr(getattr(lifecycle, "phase", None), "value", None)
    return bool(
        not getattr(page, "_closing", True)
        and not getattr(page, "_closed", True)
        and getattr(page, "_admission_state", None) is None
        and slot is not None and not slot.owned
        and callable(experiment_busy) and not experiment_busy()
        and phase == "idle"
        and getattr(lifecycle, "active_run_identity", None) is None
        and getattr(lifecycle, "attempt_run_identity", None) is None
        and controller is not None
        and not controller.browse_pending
        and not controller.viewer_1d_cleanup_pending
        and not controller.viewer_2d_cleanup_pending
    )
def epoch_token(value: float | None) -> int:
    if value is None: return 0
    if type(value) is not float or not math.isfinite(value): raise TypeError(
        "trace epoch must be a finite float or None")
    return int.from_bytes(struct.pack(">d", value), "big") + 1
def displayed_trace_input(projection: object, current: object):
    import numpy as np
    from xrd_tools.analysis.display_fit_operations import (
        DisplayedTraceInput, DisplayedTraceReceipt,
    )
    matches = tuple(trace for trace in projection.traces if trace.frame is current)
    if len(matches) != 1: raise ValueError("exact current displayed trace required")
    trace = matches[0]; frame = trace.frame; identity = frame.run_identity
    receipt = DisplayedTraceReceipt(
        "p37-display-trace-v1", identity.generation, identity.fingerprint,
        frame.source_scan, Path(frame.artifact), str(frame.local_frame_label),
        frame.work_ordinal,
    )
    return DisplayedTraceInput(
        np.array(trace.axis.values, copy=True, order="C"),
        np.array(trace.intensity, copy=True, order="C"),
        trace.axis.label, trace.axis.unit, trace.title,
        epoch_token(trace.epoch), receipt,
    )
def phase_wavelength_angstrom(controller: object, current: object) -> float | None:
    from xrd_tools.core.energy import wavelength_m_to_angstrom
    try:
        request = controller.project_request(current, require_complete=False)
        payload = controller.resolve_projection(request)
        value = None if payload is None else payload.wavelength_m
        if value is None: return None
        return wavelength_m_to_angstrom(value, allow_default_sentinel=True)
    except (AttributeError, RuntimeError, TypeError, ValueError): return None

@dataclass(frozen=True, slots=True)
class AnalysisDisplayAnchor:
    controller: object; context: object; selection: object; frame: object
    context_token: str; display_generation: int
    fingerprint: str; dialog_generation: int
def _live_anchor_facts(controller: object):
    selection = controller.selection; frame = controller.navigation.current
    if selection is None or frame is None or not controller.owns_frame(frame):
        return None
    matches = tuple(context for context in controller.projectable_contexts
                    if selection.names(context))
    if len(matches) != 1: return None
    context = matches[0]; browse = getattr(controller, "browse_context", None)
    if context is browse and not (
        context.loaded and not context.invalidated and not context.released
        and not context.commit_gate.cancelled and not controller.browse_pending
    ):
        return None
    return selection, frame, context
def capture_display_anchor(page: object, fingerprint: str,
                           dialog_generation: int) -> AnalysisDisplayAnchor | None:
    controller = page._context_controller
    facts = _live_anchor_facts(controller)
    if facts is None or type(fingerprint) is not str or type(dialog_generation) is not int: return None
    selection, frame, context = facts
    return AnalysisDisplayAnchor(
        controller, context, selection, frame, selection.context_token,
        selection.display_generation, fingerprint, dialog_generation,
    )
def display_anchor_matches(page: object, anchor: object, fingerprint: str,
                           dialog_generation: int) -> bool:
    if type(anchor) is not AnalysisDisplayAnchor or anchor.controller is not page._context_controller: return False
    facts = _live_anchor_facts(anchor.controller)
    if facts is None: return False
    selection, frame, context = facts
    return bool(
        selection is anchor.selection and frame is anchor.frame
        and context is anchor.context
        and selection.context_token == anchor.context_token
        and selection.display_generation == anchor.display_generation
        and fingerprint == anchor.fingerprint
        and dialog_generation == anchor.dialog_generation
    )
def terminal_adoption(update: object, *, current: bool) -> tuple[object | None, str]:
    from xrd_tools.analysis.scan_operations import AnalysisDisposition
    if type(update) is not OperationUpdate or update.terminal is None: return None, "P3_7_ANALYSIS_TERMINAL_INVALID"
    terminal = update.terminal
    if update.stale or not current: return None, "P3_7_ANALYSIS_STALE"
    if terminal.status is OperationTerminalStatus.FAILED: return None, terminal.diagnostic
    payload = terminal.payload
    disposition = getattr(payload, "disposition", None)
    if terminal.status is OperationTerminalStatus.CANCELLED: return None, getattr(payload, "code", "CANCELLED")
    if terminal.status is not OperationTerminalStatus.RETURNED or disposition is not AnalysisDisposition.COMPLETED:
        return None, getattr(payload, "code", "P3_7_ANALYSIS_REFUSED")
    return payload, ""
def cancel_owned(slot: object, identity: object) -> bool:
    return bool(identity is not None and slot.current_identity is identity
                and slot.cancel(identity))
def retention_admission(current: Mapping[str, object | None], replacing: str,
                        candidate: object) -> tuple[bool, str]:
    charge = getattr(candidate, "storage_bytes", None)
    if type(charge) is not int or charge < 0: return False, "P3_7_GUI_RETENTION_LIMIT"
    total = charge + sum(
        getattr(value, "storage_bytes", 0)
        for key, value in current.items() if key != replacing and value is not None
    )
    return ((True, "") if total <= RETENTION_LIMIT else
            (False, "P3_7_GUI_RETENTION_LIMIT"))
def scan_plot_plan(x: str | None, y: tuple[str, ...], normalization: str | None):
    from xrd_tools.analysis.scan_operations import ScanPlotPlan
    return ScanPlotPlan(x, tuple(y), normalization)
def metadata_plan_from_syntax(value: object):
    from xrd_tools.analysis.scan_operations import MetadataTablePlan
    from xrd_tools.core.scan import SourceSpec
    if type(value) is SourceSpec: return MetadataTablePlan(value)
    if type(value) is not tuple: return None
    if (len(value) == 2 and value[0] == "exact"
            and isinstance(value[1], (SourceSpec, str, Path)) and str(value[1])):
        return MetadataTablePlan(value[1])
    if (len(value) == 3 and value[0] == "directory"
            and isinstance(value[1], (str, Path)) and str(value[1])
            and type(value[2]) is str and value[2]):
        return MetadataTablePlan(value[1], selection="directory", kind=value[2])
    return None
def analysis_request_facts(plan: object, *, table=None, roi=None,
                           render=(), picker=0):
    from dataclasses import fields
    from xrd_tools.analysis.display_fit_operations import (
        DisplayedPeakFitPlan, DisplayedPhaseFitPlan,
    )
    from xrd_tools.analysis.scan_operations import (
        MetadataTablePlan, RoiPreviewPlan, RoiScanPlan, ScanPlotPlan,
    )
    owners = {MetadataTablePlan: "metadata", ScanPlotPlan: "scan_plot",
              RoiPreviewPlan: "roi_preview", RoiScanPlan: "roi_scan",
              DisplayedPeakFitPlan: "peak", DisplayedPhaseFitPlan: "phase"}
    kind = owners.get(type(plan))
    if kind is None or kind == "roi_scan" and plan.mask is not None: return None
    row = tuple(getattr(plan, field.name) for field in fields(plan))
    if kind in {"peak", "phase"}:
        row = (plan.trace.trace_fingerprint, plan.trace.receipt, *row[1:])
    if kind == "scan_plot":
        echo = (getattr(table, "table_fingerprint", ""),
                getattr(roi, "result_fingerprint", None), plan.x, plan.y,
                plan.normalization); row += (*echo, *render)
    elif kind in {"roi_preview", "roi_scan"}:
        echo = (plan.receipt, plan.table_fingerprint,
                plan.labels, plan.label) if kind == "roi_preview" else (
                plan.receipt, plan.table_fingerprint,
                plan.labels if plan.selected_labels is None else plan.selected_labels,
                tuple(signal.name for signal in plan.signals))
        if kind == "roi_scan": row += (picker,)
    elif kind == "metadata": echo = (plan.selection, plan.source)
    else: echo = row[:2]
    return kind, row, echo
def analysis_result_matches(payload: object, facts: object) -> bool:
    from xrd_tools.analysis.display_fit_operations import (
        DisplayedPeakFitResult, DisplayedPhaseFitResult,
    )
    from xrd_tools.analysis.scan_operations import (
        MetadataTableResult, RoiPreviewResult, RoiScanResult, ScanPlotResult,
    )
    if type(facts) is not tuple or len(facts) != 3: return False
    kind, _row, echo = facts
    if kind == "metadata":
        if type(payload) is not MetadataTableResult or payload.receipt is None: return False
        selection, source = echo
        receipt = payload.receipt
        source_spec = getattr(source, "kind", None) is not None
        identity_matches = (
            receipt.source_spec == source if source_spec
            else str(receipt.lexical_root) == str(source)
        )
        if not (payload.table_fingerprint
                and receipt.schema_version == "analysis-source-v2"
                and selection == "exact"
                and identity_matches and receipt.primary_post_state is not None):
            return False
        return True
    if kind == "scan_plot":
        table, roi, x, y, normalization = echo
        return bool(type(payload) is ScanPlotResult
                    and (payload.table_fingerprint, payload.roi_fingerprint,
                         payload.normalization) == (table, roi, normalization)
                    and (x is None or payload.x_name == x)
                    and (not y or tuple(dict.fromkeys(
                        payload.original_identities)) == y))
    if kind in {"roi_preview", "roi_scan"}:
        expected = RoiPreviewResult if kind == "roi_preview" else RoiScanResult
        if type(payload) is not expected or payload.receipt is None: return False
        suffix = ((payload.labels, payload.label) if kind == "roi_preview"
                  else (payload.requested_labels, payload.signal_names))
        return (payload.receipt, payload.table_fingerprint, *suffix) == echo
    expected = DisplayedPeakFitResult if kind == "peak" else (
        DisplayedPhaseFitResult if kind == "phase" else None)
    return bool(expected is not None and type(payload) is expected
                and (payload.trace_fingerprint, payload.trace_receipt) == echo)
def render_fit_projection(plot, residual_plot, x, result, components=()):
    import pyqtgraph as pg; from pyqtgraph.Qt import QtCore
    if result.fit is not None:
        plot.plot(x, result.fit, pen=pg.mkPen((189, 147, 249), width=2), name="fit")
    if result.background is not None:
        plot.plot(x, result.background, pen=pg.mkPen((130, 200, 160), width=1.3,
            style=QtCore.Qt.PenStyle.DashLine), name="background")
    for name, values in components: plot.plot(x, values, pen=pg.mkPen(
        width=1, style=QtCore.Qt.PenStyle.DashLine), name=name)
    for position in (() if result.marker_positions is None else result.marker_positions):
        plot.addItem(pg.InfiniteLine(float(position), angle=90,
            pen=pg.mkPen((240, 200, 90), width=1)))
    residual_plot.clear(); residual_plot.addLine(
        y=0, pen=pg.mkPen((130, 130, 140), width=1))
    if result.residual is not None:
        residual_plot.plot(x, result.residual,
                           pen=pg.mkPen((230, 133, 151), width=1))
def render_table_rows(table, headers, rows):
    table.setColumnCount(len(headers)); table.setHorizontalHeaderLabels(headers)
    table.setRowCount(len(rows))
    for row, values in enumerate(rows):
        for column, value in enumerate(values):
            table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
class MetadataResultDialog(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Metadata"); self.setObjectName("p37MetadataDialog")
        layout = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget(0, 0, self); self.status = QtWidgets.QLabel("")
        layout.addWidget(self.table); layout.addWidget(self.status)
    def clear_result(self) -> None:
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.status.clear()
    def adopt_result(self, result: object) -> None:
        columns = tuple(getattr(result, "columns", ()))
        labels = tuple(getattr(result, "labels", ()))
        self.table.setColumnCount(len(columns)); self.table.setRowCount(len(labels))
        self.table.setHorizontalHeaderLabels([column.name for column in columns])
        for col, column in enumerate(columns):
            values = column.numeric if column.numeric is not None else column.text
            for row, value in enumerate(() if values is None else values):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(
                    "" if value is None else str(value)))
        self.status.setText(getattr(result, "code", ""))
