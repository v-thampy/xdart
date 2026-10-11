"""Passive scientific plot options and the production waterfall activation policy."""

from __future__ import annotations

from dataclasses import replace

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .scientific_waterfall_policy import waterfall_should_be_active
from .shell_values import ScientificPlotOptions


WATERFALL_Y_AXIS_CHOICES = (
    "Frame #",
    "Time (s)",
    "Time (minutes)",
)
INTENSITY_SCALE_CHOICES = ("Linear", "Sqrt", "Log")
COVERAGE_TOOLTIP = (
    "Shade Q–χ bins that received no contributing pixels. Coverage includes pixel splitting "
    "and does not measure angular completeness."
)
RING_TOOLTIP = ("Overlay predicted powder rings using this frame's effective run geometry and wavelength. "
                'Radial agreement alone does not identify detector mounting.')
GRID_TOOLTIP = ('Show detector-frame q contours and χ half-rays with the effective Run plan’s χ offset. '
                'Display only; available on Standard Int Run frames with a uniform flat detector.')

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
    (("other", "coverage_holes"), "show_coverage_holes"),
    (("other", "calibrant"), "calibrant"),
    (("other", "coordinate_grid"), "coordinate_grid"),
    (("other", "phase_references"), "phase_references"),
    (("other", "phase_min_intensity"), "phase_min_intensity"),
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
        self.setWindowTitle("Plot Options")
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

        support = self._section(layout, "2-D Support")
        support.parentWidget().setToolTip(COVERAGE_TOOLTIP)
        self.show_coverage_holes = QtWidgets.QCheckBox("Shade uncovered bins")
        support.addWidget(self.show_coverage_holes, 0, 0)
        self.set_coverage_available(False)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        self.accept_button = QtWidgets.QPushButton("Okay")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        buttons.addWidget(self.accept_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self.accept_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        from xrd_tools.integrate.calibration import CALIBRANT_NAMES
        self.calibrant = self.add_choice('Calibrant', ('Off', *CALIBRANT_NAMES), 'Off', tooltip=RING_TOOLTIP)
        self.set_ring_available(False, visible=False)  # shown only when Int's raw display reconciles
        self.coordinate_grid = self.add_choice('Coordinate grid', ('Off', 'q', 'χ', 'q + χ'),
                                               'Off', tooltip=GRID_TOOLTIP)
        self.set_grid_available(False, visible=False)

        from xdart.gui.analysis.phase_source import PhaseSourceWidget
        from xdart.gui.widgets.controls_panel import FormRow
        phase = self._section(layout, 'Reference phases')
        self._phase_group = phase.parentWidget()
        self._phase_group.setToolTip('Choose reference phases for q/2θ lines and current-Run detector rings. Display only.')
        self.phase_source = PhaseSourceWidget(self._phase_group)
        phase.addWidget(self.phase_source, 0, 0)
        self.phase_floor = FormRow(label='Minimum intensity (%)', path=('other', 'phase_min_intensity'),
                                   value='5', parent=self._phase_group)
        self.phase_floor.editor.setValidator(QtGui.QDoubleValidator(0., 100., 6, self.phase_floor))
        phase.addWidget(self.phase_floor, 1, 0)
        self._phase_available = True
        self.phase_source.changed.connect(lambda _value: self._sync_phase_floor())
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

    def mount_display_scale(self, controls: QtWidgets.QWidget) -> None:
        """Mount the view's existing live display controls above Okay/Cancel."""
        layout = self.layout()
        layout.insertWidget(layout.count() - 1, controls)

    def set_coverage_available(self, available: bool) -> None:
        self.show_coverage_holes.setEnabled(available)
        self.show_coverage_holes.setToolTip(
            COVERAGE_TOOLTIP if available else COVERAGE_TOOLTIP +
            " Available on a Stitch Q–χ intensity map with stored coverage."
        )

    def add_choice(self, label: str, choices: tuple[str, ...], value: str, *, tooltip: str = '') -> QtWidgets.QComboBox:
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
        caption = QtWidgets.QLabel(label)
        caption.setToolTip(tooltip); editor.setToolTip(tooltip); row.setToolTip(tooltip)
        layout.addWidget(caption)
        layout.addWidget(editor)
        outer = self.layout()
        outer.insertWidget(outer.count() - 1, row)
        return editor

    def set_ring_available(self, available: bool, reason: str = '', *, visible: bool = True) -> None:
        from xrd_tools.geometry_figures import RUN_RING_REASON
        tip = RING_TOOLTIP if available else RING_TOOLTIP+' '+(reason or RUN_RING_REASON)
        self.calibrant.setEnabled(available)
        self.calibrant.setToolTip(tip)
        row = self.calibrant.parentWidget(); row.setVisible(visible); row.setToolTip(tip)
        row.findChild(QtWidgets.QLabel).setToolTip(tip)

    def set_grid_available(self, available: bool, reason: str = '', *, visible: bool = True) -> None:
        from xrd_tools.geometry_figures import RUN_RING_REASON
        tip = GRID_TOOLTIP if available else GRID_TOOLTIP+' '+(reason or RUN_RING_REASON)
        self.coordinate_grid.setEnabled(available)
        row = self.coordinate_grid.parentWidget()
        row.setVisible(visible)
        for widget in (row, self.coordinate_grid, row.findChild(QtWidgets.QLabel)):
            widget.setToolTip(tip)

    def set_phase_available(self, available, reason='', *, visible=True):
        self._phase_available = bool(available)
        self._phase_group.setVisible(visible)
        self.phase_source.set_available(available, reason)
        self._sync_phase_floor(reason)

    def _sync_phase_floor(self, reason=''):
        entries = self.phase_source.selected_entries()
        cif = any(entry.source_kind == 'cif' for entry in entries)
        available = self._phase_available and cif
        tip = ('Minimum intensity as a percent of the strongest reflection in the reachable range. '
               'Applies to CIF phases only; lattice phases have no structural intensities.')
        if not available:
            tip += ' ' + (reason or ('Lattice-only phases bypass the floor.' if entries else 'Select a CIF phase to edit the floor.'))
        self.phase_floor.editor.setEnabled(available)
        self.phase_floor.set_purpose_tooltip(tip)

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
            self.show_coverage_holes,
            self.calibrant,
            self.coordinate_grid,
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
        self.show_coverage_holes.setChecked(options.show_coverage_holes)
        self.calibrant.setCurrentText(options.calibrant)
        self.coordinate_grid.setCurrentText(options.coordinate_grid)
        source_block = QtCore.QSignalBlocker(self.phase_source)
        self.phase_source.set_selection(options.phase_references)
        self.phase_floor.editor.setText(str(options.phase_min_intensity))
        self._projection = options
        self._sync_phase_floor()
        del source_block, blockers

    def edited_options(self) -> ScientificPlotOptions:
        return ScientificPlotOptions(
            waterfall_y_axis=self.waterfall_y_axis.currentText(),
            waterfall_start=self.waterfall_start.value(),
            waterfall_stop=self.waterfall_stop.value(),
            waterfall_step=self.waterfall_step.value(),
            overlay_offset=float(self.overlay_offset.value()),
            show_legend=self.show_legend.isChecked(),
            intensity_scale=self.intensity_scale.currentText(),
            show_coverage_holes=self.show_coverage_holes.isChecked(),
            calibrant=self.calibrant.currentText(),
            coordinate_grid=self.coordinate_grid.currentText(),
            phase_references=self.phase_source.selection,
            phase_min_intensity=float(self.phase_floor.editor.text()),
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
