"""Passive 1-D plot options and the production waterfall activation policy."""

from __future__ import annotations

from dataclasses import replace

from pyqtgraph.Qt import QtCore, QtWidgets

from .scientific_waterfall_policy import waterfall_should_be_active
from .shell_values import ScientificPlotOptions


WATERFALL_Y_AXIS_CHOICES = (
    "Frame #",
    "Time (s)",
    "Time (minutes)",
)
INTENSITY_SCALE_CHOICES = ("Linear", "Sqrt", "Log")

#: Every 1-D plot option: its command path and its ``ScientificPlotOptions`` field, the one table the
#: view emits and every page applies (Codex general audit AUD-8).
PLOT_OPTION_BINDINGS = (
    (("waterfall", "y_axis"), "waterfall_y_axis"),
    (("waterfall", "start"), "waterfall_start"),
    (("waterfall", "stop"), "waterfall_stop"),
    (("waterfall", "step"), "waterfall_step"),
    (("overlay", "offset"), "overlay_offset"),
    (("other", "legend"), "show_legend"),
    (("other", "intensity_scale"), "intensity_scale"),
)


def plot_option_update(options: ScientificPlotOptions, path, value) -> ScientificPlotOptions | None:
    """``options`` with the option at command ``path`` set to ``value``; None for an unknown path or a
    value the option refuses (``ScientificPlotOptions`` validates each; an integer offset is a float)."""
    field = dict(PLOT_OPTION_BINDINGS).get(path)
    if field is None:
        return None
    if field == "overlay_offset" and type(value) is int:
        value = float(value)
    try:
        return replace(options, **{field: value})
    except (TypeError, ValueError):
        return None


class PlotOptionsDialog(QtWidgets.QDialog):
    """A projection-reconciled editor with no scientific state ownership."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        show_waterfall: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("e4PlotOptionsDialog")
        self.setWindowTitle("1-D Plot Options")
        self.setModal(False)
        self._projection = ScientificPlotOptions()

        layout = QtWidgets.QVBoxLayout(self)
        waterfall = self._section(layout, "Waterfall")
        self._waterfall_group = waterfall.parentWidget()
        self._waterfall_group.setVisible(show_waterfall)
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

    def set_waterfall_visible(self, visible: bool) -> None:
        self._waterfall_group.setVisible(bool(visible))

    def add_choice(self, label: str, choices: tuple[str, ...], value: str) -> QtWidgets.QComboBox:
        """A host's own choice among ``choices``, built here above the buttons (Codex's Stitch/Int reuse
        audit R3: Stitch's Diagnostic was a combo box the page built). Returns the editor; the host
        applies its value on ``ScientificView.plotOptionsAccepted`` and resets it on ``rejected``."""
        if not choices or value not in choices:
            raise ValueError("a plot-options choice needs its choices and a value among them")
        editor = QtWidgets.QComboBox()
        editor.setObjectName("e4PlotOptionsHostChoice")
        editor.addItems(choices)
        editor.setCurrentText(value)
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel(label))
        layout.addWidget(editor)
        outer = self.layout()
        outer.insertWidget(outer.count() - 1, row)
        return editor

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
    "PLOT_OPTION_BINDINGS",
    "PlotOptionsDialog",
    "WATERFALL_Y_AXIS_CHOICES",
    "plot_option_update",
    "waterfall_should_be_active",
]
