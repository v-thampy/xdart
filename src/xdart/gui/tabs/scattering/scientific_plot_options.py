"""Passive 1-D plot options and the production waterfall activation policy."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from .shell_values import ScientificPlotOptions


WATERFALL_Y_AXIS_CHOICES = (
    "Frame #",
    "Time (s)",
    "Time (minutes)",
)
INTENSITY_SCALE_CHOICES = ("Linear", "Sqrt", "Log")

PLOT_OPTION_COMMAND_PATHS = (
    ("waterfall", "y_axis"),
    ("waterfall", "start"),
    ("waterfall", "stop"),
    ("waterfall", "step"),
    ("overlay", "offset"),
    ("other", "legend"),
    ("other", "intensity_scale"),
)


def waterfall_should_be_active(
    plot_mode: str,
    trace_count: int,
    *,
    was_active: bool,
) -> bool:
    """Return the exact production bottom-panel waterfall state.

    Explicit Waterfall starts on the fourth trace. Overlay and legacy
    multi-selected Single start on the sixteenth, then retain the image view
    through eight traces and return to curves at seven. Aggregate modes never
    use the waterfall view.
    """

    count = max(0, int(trace_count))
    if plot_mode == "Waterfall":
        return count >= 4
    if plot_mode in {"Average", "Sum"}:
        return False
    if plot_mode in {"Overlay", "Single"}:
        return count >= (8 if was_active else 16)
    return False


class PlotOptionsDialog(QtWidgets.QDialog):
    """A projection-reconciled editor with no scientific state ownership."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("e4PlotOptionsDialog")
        self.setWindowTitle("1-D Plot Options")
        self.setModal(False)
        self._projection = ScientificPlotOptions()

        layout = QtWidgets.QVBoxLayout(self)
        waterfall = self._section(layout, "Waterfall")
        waterfall.addWidget(QtWidgets.QLabel("Y-Axis"), 0, 0)
        waterfall.addWidget(QtWidgets.QLabel("Start"), 0, 1)
        waterfall.addWidget(QtWidgets.QLabel("Stop"), 0, 2)
        waterfall.addWidget(QtWidgets.QLabel("Step"), 0, 3)
        self.waterfall_y_axis = QtWidgets.QComboBox()
        self.waterfall_y_axis.addItems(WATERFALL_Y_AXIS_CHOICES)
        self.waterfall_start = self._integer_spin(1, 100_000, 1)
        self.waterfall_stop = self._integer_spin(0, 100_000, 0)
        self.waterfall_stop.setSpecialValueText("End")
        self.waterfall_step = self._integer_spin(1, 1_000, 1)
        waterfall.addWidget(self.waterfall_y_axis, 1, 0)
        waterfall.addWidget(self.waterfall_start, 1, 1)
        waterfall.addWidget(self.waterfall_stop, 1, 2)
        waterfall.addWidget(self.waterfall_step, 1, 3)

        overlay = self._section(layout, "Overlay")
        overlay.addWidget(QtWidgets.QLabel("Offset"), 0, 0)
        self.overlay_offset = QtWidgets.QDoubleSpinBox()
        self.overlay_offset.setDecimals(1)
        self.overlay_offset.setRange(-1_000_000.0, 1_000_000.0)
        self.overlay_offset.setSingleStep(5.0)
        overlay.addWidget(self.overlay_offset, 0, 1)

        other = self._section(layout, "Other")
        self.show_legend = QtWidgets.QPushButton("Legend")
        self.show_legend.setCheckable(True)
        self.intensity_scale = QtWidgets.QComboBox()
        self.intensity_scale.addItems(INTENSITY_SCALE_CHOICES)
        other.addWidget(self.show_legend, 0, 0)
        other.addWidget(self.intensity_scale, 0, 1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        self.accept_button = QtWidgets.QPushButton("Okay")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        buttons.addWidget(self.accept_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self.accept_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        self.reconcile(self._projection)

    @staticmethod
    def _section(
        outer: QtWidgets.QVBoxLayout,
        title: str,
    ) -> QtWidgets.QGridLayout:
        group = QtWidgets.QGroupBox(title)
        grid = QtWidgets.QGridLayout(group)
        outer.addWidget(group)
        return grid

    @staticmethod
    def _integer_spin(
        minimum: int,
        maximum: int,
        value: int,
    ) -> QtWidgets.QSpinBox:
        widget = QtWidgets.QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    @property
    def projection(self) -> ScientificPlotOptions:
        return self._projection

    def reconcile(self, options: ScientificPlotOptions) -> None:
        if type(options) is not ScientificPlotOptions:
            raise TypeError("plot options must be an exact frozen value")
        widgets = (
            self.waterfall_y_axis,
            self.waterfall_start,
            self.waterfall_stop,
            self.waterfall_step,
            self.overlay_offset,
            self.show_legend,
            self.intensity_scale,
        )
        blockers = [QtCore.QSignalBlocker(widget) for widget in widgets]
        self.waterfall_y_axis.clear()
        self.waterfall_y_axis.addItems(WATERFALL_Y_AXIS_CHOICES)
        if self.waterfall_y_axis.findText(options.waterfall_y_axis) < 0:
            self.waterfall_y_axis.addItem(options.waterfall_y_axis)
        self.waterfall_y_axis.setCurrentText(options.waterfall_y_axis)
        self.waterfall_start.setValue(options.waterfall_start)
        self.waterfall_stop.setValue(options.waterfall_stop)
        self.waterfall_step.setValue(options.waterfall_step)
        self.overlay_offset.setValue(options.overlay_offset)
        self.show_legend.setChecked(options.show_legend)
        self.intensity_scale.setCurrentText(options.intensity_scale)
        self._projection = options
        del blockers

    def edited_options(self) -> ScientificPlotOptions:
        return ScientificPlotOptions(
            waterfall_y_axis=self.waterfall_y_axis.currentText(),
            waterfall_start=self.waterfall_start.value(),
            waterfall_stop=self.waterfall_stop.value(),
            waterfall_step=self.waterfall_step.value(),
            overlay_offset=float(self.overlay_offset.value()),
            show_legend=self.show_legend.isChecked(),
            intensity_scale=self.intensity_scale.currentText(),
        )

    def reject(self) -> None:
        self.reconcile(self._projection)
        super().reject()


__all__ = [
    "INTENSITY_SCALE_CHOICES",
    "PLOT_OPTION_COMMAND_PATHS",
    "PlotOptionsDialog",
    "WATERFALL_Y_AXIS_CHOICES",
    "waterfall_should_be_active",
]
