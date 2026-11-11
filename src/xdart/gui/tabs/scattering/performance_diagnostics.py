"""Config ▸ Performance Diagnostics: timing, the Live plot interval and next-Run pipeline values.

The window is the shared Advanced dialog (``AdvancedDialog``) with Controls rows (``FormRow``), as every settings
dialog is (hard rule 14; 1.2.10 review M3). Its pipeline and output values apply only to a Run that can take them, a
non-Live, non-Batch Overwrite Run with NeXus output; any other Run ignores them and says so once
(``adapters/dynamic_output.py``), and the session never keeps them (``run_intent_profile``). With nothing stored the
window shows what such a Run uses without them, the adapter's own defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping

from pyqtgraph.Qt import QtWidgets

from xrd_tools.session.intent_store import RunIntentSnapshot
from xdart.gui.widgets.controls_panel import FormRow

from .adapters.dynamic_output import DEFAULT_OUTPUT_DIAGNOSTICS, DEFAULT_PIPELINE_V2
from .advanced_editor import AdvancedDialog


_PIPELINE_KEY = "_post_g2_pipeline_v2"
_OUTPUT_DIAGNOSTICS_KEY = "_post_g2_output_diagnostics_v1"
_PIPELINE_FIELDS = (
    "writer_settlement_batch_size",
    "nexus_record_batch_size",
    "reduction_inflight",
    "semantic_checkpoint_frame_cap",
    "staging_frame_cap",
)
_DEFAULT_PIPELINE = tuple(getattr(DEFAULT_PIPELINE_V2, field) for field in _PIPELINE_FIELDS)
_DEFAULT_OUTPUT = (DEFAULT_OUTPUT_DIAGNOSTICS.save_xye, DEFAULT_OUTPUT_DIAGNOSTICS.durable_fsync)
_PATH = "PerformanceDiagnostics"


@dataclass(frozen=True, slots=True)
class PerformanceDiagnosticsValues:
    settlement: int
    record: int
    inflight: int
    checkpoint: int
    plot_interval_ms: int
    save_xye: bool = DEFAULT_OUTPUT_DIAGNOSTICS.save_xye
    durable_fsync: bool = DEFAULT_OUTPUT_DIAGNOSTICS.durable_fsync
    staging_frame_cap: int = DEFAULT_PIPELINE_V2.staging_frame_cap
    quartile_telemetry: bool = False

    def pipeline_mapping(self) -> dict[str, int]:
        return dict(zip(
            _PIPELINE_FIELDS,
            (
                self.settlement,
                self.record,
                self.inflight,
                self.checkpoint,
                self.staging_frame_cap,
            ),
            strict=True,
        ))

    def output_diagnostics_mapping(self) -> dict[str, bool]:
        return {
            "save_xye": self.save_xye,
            "durable_fsync": self.durable_fsync,
        }


def performance_diagnostics_are_defaults(values: PerformanceDiagnosticsValues) -> bool:
    """True when the pipeline and output values are what a Run uses with nothing stored, so Apply stores none."""
    return (tuple(values.pipeline_mapping()[field] for field in _PIPELINE_FIELDS) == _DEFAULT_PIPELINE
            and (values.save_xye, values.durable_fsync) == _DEFAULT_OUTPUT)


def performance_diagnostics_error(
    values: PerformanceDiagnosticsValues,
) -> str:
    if type(values) is not PerformanceDiagnosticsValues:
        return "Performance diagnostics values are invalid."
    if (
        type(values.save_xye) is not bool
        or type(values.durable_fsync) is not bool
        or type(values.quartile_telemetry) is not bool
    ):
        return "Output diagnostics values must be exact booleans."
    row = (
        values.settlement,
        values.record,
        values.inflight,
        values.checkpoint,
        values.plot_interval_ms,
        values.staging_frame_cap,
    )
    if any(type(value) is not int for value in row):
        return "Performance diagnostics values must be exact integers."
    settlement, record, inflight, checkpoint, plot, staging = row
    if not (
        1 <= settlement <= 16
        and 1 <= record <= 16
        and 1 <= inflight <= 64
        and 1 <= checkpoint <= 10_000
        and plot >= 125
        and 9 <= staging <= 10_008
    ):
        return "Performance diagnostics values are invalid or out of bounds."
    if settlement > inflight or record > checkpoint or checkpoint % settlement:
        return "Performance diagnostics values violate batching bounds."
    if checkpoint > staging - 8:
        return (
            "Semantic checkpoint exceeds the staging cap minus its "
            "8-frame safety margin."
        )
    return ""


def _loaded_pipeline(
    snapshot: RunIntentSnapshot,
) -> tuple[int, int, int, int, int]:
    candidate = snapshot.thaw().run_options.get(_PIPELINE_KEY)
    if not isinstance(candidate, Mapping):
        return _DEFAULT_PIPELINE
    if set(candidate) != set(_PIPELINE_FIELDS):
        return _DEFAULT_PIPELINE
    row = tuple(candidate[field] for field in _PIPELINE_FIELDS)
    if any(type(value) is not int for value in row):
        return _DEFAULT_PIPELINE
    values = PerformanceDiagnosticsValues(
        *row[:4], 125, staging_frame_cap=row[4],
    )
    return row if not performance_diagnostics_error(values) else _DEFAULT_PIPELINE


def _loaded_output_diagnostics(
    snapshot: RunIntentSnapshot,
) -> tuple[bool, bool]:
    candidate = snapshot.thaw().run_options.get(_OUTPUT_DIAGNOSTICS_KEY)
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "save_xye", "durable_fsync",
    }:
        return _DEFAULT_OUTPUT
    row = candidate["save_xye"], candidate["durable_fsync"]
    return row if all(type(value) is bool for value in row) else _DEFAULT_OUTPUT


def _label(text: str, tip: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setToolTip(tip)
    return label


class PerformanceDiagnosticsDialog(AdvancedDialog):
    """Int's Advanced dialog with the diagnostics rows; its values apply only on Apply (``edit``)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent, title="Performance Diagnostics",
                         object_name="vnextPerformanceDiagnosticsDialog", accept_buttons=True)
        self.setModal(True)
        settlement, record, inflight, checkpoint, staging = _DEFAULT_PIPELINE
        self.note = _label(
            "Timing (Config ▸ Debug ▸ Performance Monitoring) and the Live plot interval apply to every Run at "
            "once. The pipeline and output values apply only to a non-Live, non-Batch Overwrite Run with NeXus "
            "output; any other Run ignores them and says so once in the log. Values equal to the defaults store "
            "nothing.",
            "What Apply changes, and which Runs use it.")
        self.add_section(self.note)
        self.settlement = self._int("settlement", "Settlement batch", settlement, 1, 16,
                                    "Frames the writer settles together (1 to 16, at most Reduction in-flight; the "
                                    f"Semantic checkpoint is a multiple of it). Default {settlement}.")
        self.record = self._int("record", "NeXus record batch", record, 1, 16,
                                "Frames written to the NeXus file in one call (1 to 16, at most the Semantic "
                                f"checkpoint). Default {record}.")
        self.inflight = self._int("inflight", "Reduction in-flight", inflight, 1, 64,
                                  f"Frames being reduced at once (1 to 64). Default {inflight}.")
        self.checkpoint = self._int("checkpoint", "Semantic checkpoint", checkpoint, 1, 10_000,
                                    "Frames between checkpoints that make the output readable and recoverable "
                                    f"(1 to 10000, at most the Staging cap minus 8). Default {checkpoint}.")
        self.staging = self._int("staging", "Staging cap", staging, 9, 10_008,
                                 "Frames held before they are written (9 to 10008). Large caps let memory grow "
                                 "with the frame count and can use tens of GiB; resources are admitted exactly at "
                                 f"Run. Default {staging}.")
        self.add_group("Pipeline (next Overwrite Run)", rows=(
            self.settlement, self.record, self.inflight, self.checkpoint, self.staging))
        self.save_xye = self._bool("save_xye", "Save XYE sidecars", _DEFAULT_OUTPUT[0],
                                   "Write an .xye file beside each frame's NeXus row (next Overwrite Run). On by "
                                   "default.")
        self.durable_fsync = self._bool("durable_fsync", "Durable fsync", _DEFAULT_OUTPUT[1],
                                        "Off by default: a processed file can always be regenerated, so checkpoints "
                                        "and the close are not forced to disk. On syncs each checkpoint and the "
                                        "close to disk (next Overwrite Run), which is slower on network drives.")
        self.add_group("Output (next Overwrite Run)", rows=(self.save_xye, self.durable_fsync))
        self.plot_interval = self._int("plot_interval", "Live plot interval (ms)", 250, 125, 60_000,
                                       "How often a running Run redraws its plots, in milliseconds (at least 125). "
                                       "Applies at once, in every Run mode.")
        self.quartile_telemetry = self._bool("quartile_telemetry", "Within-run quartile timing", False,
                                             "Add timing for each quarter of the next Run to the timing summary.")
        self.add_group("Timing and display (every Run)", rows=(self.plot_interval, self.quartile_telemetry))
        self.fsync_default = _label(
            "Durable fsync is off by default; turn it on only to measure its cost.",
            "A processed file can always be regenerated from its raw frames.")
        self.add_section(self.fsync_default)
        self.refusal = _label("", "Why Apply was refused.")
        self.refusal.hide()
        self.add_section(self.refusal)
        apply = self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        apply.setText("Apply")
        apply.setToolTip("Use these values: timing and the plot interval now, pipeline and output values on the next "
                         "Run that can take them.")
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setToolTip(
            "Close without changing anything.")

    def _int(self, name, label, value, minimum, maximum, tip) -> FormRow:
        row = FormRow(label=label, path=(_PATH, name), value=value, kind="int", minimum=minimum, maximum=maximum)
        row.set_purpose_tooltip(tip)
        return row

    def _bool(self, name, label, value, tip) -> FormRow:
        row = FormRow(label=label, path=(_PATH, name), value=value, kind="bool")
        row.set_purpose_tooltip(tip)
        return row

    def _numbers(self) -> tuple[FormRow, ...]:
        return (self.settlement, self.record, self.inflight, self.checkpoint, self.staging, self.plot_interval)

    def load(self, snapshot: RunIntentSnapshot, plot_interval_ms: int) -> None:
        """Show the stored values, or the defaults where none is stored."""
        for row, value in zip(
            (self.settlement, self.record, self.inflight, self.checkpoint, self.staging),
            _loaded_pipeline(snapshot),
            strict=True,
        ):
            row.set_number(value)
        self.plot_interval.set_number(max(125, int(plot_interval_ms)))
        save_xye, durable_fsync = _loaded_output_diagnostics(snapshot)
        self.save_xye.editor.setChecked(save_xye)
        self.durable_fsync.editor.setChecked(durable_fsync)
        self.quartile_telemetry.editor.setChecked(os.environ.get("XDART_PERF_QUARTILES", "").strip() == "1")
        self.refusal.hide()

    def values(self) -> PerformanceDiagnosticsValues:
        return PerformanceDiagnosticsValues(
            self.settlement.editor.value(),
            self.record.editor.value(),
            self.inflight.editor.value(),
            self.checkpoint.editor.value(),
            self.plot_interval.editor.value(),
            self.save_xye.current_value(),
            self.durable_fsync.current_value(),
            self.staging.editor.value(),
            self.quartile_telemetry.current_value(),
        )

    def accept(self) -> None:
        """Apply only whole numbers within their bounds; a refused row says why and keeps the window open."""
        refused = [row.refusal() for row in self._numbers() if row.refusal()]
        if refused:
            self.refusal.setText(refused[0])
            self.refusal.show()
            return
        super().accept()

    def edit(
        self,
        snapshot: RunIntentSnapshot,
        plot_interval_ms: int,
    ) -> PerformanceDiagnosticsValues | None:
        self.load(snapshot, plot_interval_ms)
        if self.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return self.values()


__all__ = [
    "PerformanceDiagnosticsDialog",
    "PerformanceDiagnosticsValues",
    "performance_diagnostics_are_defaults",
    "performance_diagnostics_error",
]
