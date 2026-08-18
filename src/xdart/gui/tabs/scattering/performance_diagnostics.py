"""Explicit, next-run controls for the private post-G2 performance probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from pyqtgraph.Qt import QtWidgets

from xrd_tools.session.intent_store import RunIntentSnapshot


_PIPELINE_KEY = "_post_g2_pipeline_v2"
_OUTPUT_DIAGNOSTICS_KEY = "_post_g2_output_diagnostics_v1"
_PIPELINE_FIELDS = (
    "writer_settlement_batch_size",
    "nexus_record_batch_size",
    "reduction_inflight",
    "semantic_checkpoint_frame_cap",
)
_DEFAULT_PIPELINE = (1, 8, 16, 56)


@dataclass(frozen=True, slots=True)
class PerformanceDiagnosticsValues:
    settlement: int
    record: int
    inflight: int
    checkpoint: int
    plot_interval_ms: int
    save_xye: bool = True
    durable_fsync: bool = True

    def pipeline_mapping(self) -> dict[str, int]:
        return dict(zip(
            _PIPELINE_FIELDS,
            (
                self.settlement,
                self.record,
                self.inflight,
                self.checkpoint,
            ),
            strict=True,
        ))

    def output_diagnostics_mapping(self) -> dict[str, bool]:
        return {
            "save_xye": self.save_xye,
            "durable_fsync": self.durable_fsync,
        }


def performance_diagnostics_error(
    values: PerformanceDiagnosticsValues,
) -> str:
    if type(values) is not PerformanceDiagnosticsValues:
        return "Performance diagnostics values are invalid."
    if type(values.save_xye) is not bool or type(values.durable_fsync) is not bool:
        return "Output diagnostics values must be exact booleans."
    row = (
        values.settlement,
        values.record,
        values.inflight,
        values.checkpoint,
        values.plot_interval_ms,
    )
    if any(type(value) is not int for value in row):
        return "Performance diagnostics values must be exact integers."
    settlement, record, inflight, checkpoint, plot = row
    if not (
        1 <= settlement <= 16
        and 1 <= record <= 16
        and 1 <= inflight <= 64
        and checkpoint >= 1
        and plot >= 125
    ):
        return "Performance diagnostics values are invalid or out of bounds."
    if settlement > inflight or record > checkpoint or checkpoint % settlement:
        return "Performance diagnostics values violate batching bounds."
    return ""


def _loaded_pipeline(snapshot: RunIntentSnapshot) -> tuple[int, int, int, int]:
    candidate = snapshot.thaw().run_options.get(_PIPELINE_KEY)
    if not isinstance(candidate, Mapping) or set(candidate) != set(_PIPELINE_FIELDS):
        return _DEFAULT_PIPELINE
    row = tuple(candidate[field] for field in _PIPELINE_FIELDS)
    if any(type(value) is not int for value in row):
        return _DEFAULT_PIPELINE
    values = PerformanceDiagnosticsValues(*row, 125)
    return row if not performance_diagnostics_error(values) else _DEFAULT_PIPELINE


def _loaded_output_diagnostics(
    snapshot: RunIntentSnapshot,
) -> tuple[bool, bool]:
    candidate = snapshot.thaw().run_options.get(_OUTPUT_DIAGNOSTICS_KEY)
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "save_xye", "durable_fsync",
    }:
        return True, True
    row = candidate["save_xye"], candidate["durable_fsync"]
    return row if all(type(value) is bool for value in row) else (True, True)


class PerformanceDiagnosticsDialog(QtWidgets.QDialog):
    """Compact editor whose defaults are inert until explicit acceptance."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("vnextPerformanceDiagnosticsDialog")
        self.setWindowTitle("Performance Diagnostics")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Pipeline values apply to the next Run. Plot cadence applies "
            "immediately. Pipeline tuning requires Standard non-Live, "
            "non-Batch, Overwrite, coordinated NeXus output; incompatible "
            "choices fail admission."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QtWidgets.QFormLayout()
        self.settlement = self._spin("performanceSettlement", 1, 16)
        self.record = self._spin("performanceRecord", 1, 16)
        self.inflight = self._spin("performanceInflight", 1, 64)
        self.checkpoint = self._spin("performanceCheckpoint", 1, 1_000_000)
        self.plot_interval = self._spin("performancePlotInterval", 125, 60_000)
        self.save_xye = QtWidgets.QCheckBox("Save XYE sidecars")
        self.save_xye.setObjectName("performanceSaveXye")
        self.durable_fsync = QtWidgets.QCheckBox("Durable fsync")
        self.durable_fsync.setObjectName("performanceDurableFsync")
        self.plot_interval.setSuffix(" ms")
        form.addRow("Settlement batch", self.settlement)
        form.addRow("NeXus record batch", self.record)
        form.addRow("Reduction in-flight", self.inflight)
        form.addRow("Semantic checkpoint", self.checkpoint)
        form.addRow("Live plot interval", self.plot_interval)
        form.addRow("Output", self.save_xye)
        form.addRow("Safety", self.durable_fsync)
        layout.addLayout(form)
        warning = QtWidgets.QLabel(
            "Turning off Durable fsync is a diagnostic benchmark only; "
            "crash or power-loss persistence is not guaranteed."
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        ).setText("Apply")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _spin(name: str, minimum: int, maximum: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setObjectName(name)
        spin.setRange(minimum, maximum)
        return spin

    def edit(
        self,
        snapshot: RunIntentSnapshot,
        plot_interval_ms: int,
    ) -> PerformanceDiagnosticsValues | None:
        pipeline = _loaded_pipeline(snapshot)
        save_xye, durable_fsync = _loaded_output_diagnostics(snapshot)
        for editor, value in zip(
            (self.settlement, self.record, self.inflight, self.checkpoint),
            pipeline,
            strict=True,
        ):
            editor.setValue(value)
        self.plot_interval.setValue(max(125, int(plot_interval_ms)))
        self.save_xye.setChecked(save_xye)
        self.durable_fsync.setChecked(durable_fsync)
        if self.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return PerformanceDiagnosticsValues(
            self.settlement.value(),
            self.record.value(),
            self.inflight.value(),
            self.checkpoint.value(),
            self.plot_interval.value(),
            self.save_xye.isChecked(),
            self.durable_fsync.isChecked(),
        )


__all__ = [
    "PerformanceDiagnosticsDialog",
    "PerformanceDiagnosticsValues",
    "performance_diagnostics_error",
]
