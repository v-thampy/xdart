"""Native value editor for revision-owned integration settings."""

from __future__ import annotations

from dataclasses import dataclass

from pyqtgraph.Qt import QtCore, QtWidgets

from xrd_tools.session.intent_store import RunIntentSnapshot

from .controls_editing import (
    ADVANCED_METHODS_1D,
    ADVANCED_METHODS_2D,
    GI_HISTOGRAM_METHODS,
    AdvancedDimensionValues,
    AdvancedSettingsValues,
    advanced_settings_values,
)
from xrd_tools.integrate.calibration import PONI_V3_SENSOR_MATERIALS


@dataclass(slots=True)
class _DimensionEditors:
    solid_angle: QtWidgets.QCheckBox
    apply_polarization: QtWidgets.QCheckBox
    polarization_factor: QtWidgets.QDoubleSpinBox
    method: QtWidgets.QComboBox
    dummy: QtWidgets.QLineEdit
    delta_dummy: QtWidgets.QLineEdit
    chi_offset: QtWidgets.QLineEdit
    safe: QtWidgets.QCheckBox


class AdvancedSettingsDialog(QtWidgets.QDialog):
    """A combined 1D/2D editor containing values, not legacy widgets."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("vnextAdvancedSettingsDialog")
        self.setWindowTitle("Integration — Advanced Settings")
        self.setModal(True)
        self._gi_enabled: object = False
        layout = QtWidgets.QVBoxLayout(self)
        self.gi_group = QtWidgets.QGroupBox(
            "Grazing-incidence Fiber integration"
        )
        self.gi_group.setObjectName("vnextGiFiberAdvancedGroup")
        gi_form = QtWidgets.QFormLayout(self.gi_group)
        self.gi_method = QtWidgets.QComboBox()
        self.gi_method.setObjectName("vnextGiHistogramBackend")
        choices = {
            "cython": (
                "Cython histogram (fast, default)",
                "Compiled pyFAI Fiber histogram backend.",
            ),
            "python": (
                "Python histogram (reference)",
                "Python pyFAI Fiber histogram backend for comparison.",
            ),
        }
        for method in GI_HISTOGRAM_METHODS:
            label, tooltip = choices[method]
            self.gi_method.addItem(label, method)
            self.gi_method.setItemData(
                self.gi_method.count() - 1,
                tooltip,
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        self.gi_method.setToolTip(
            "Select the histogram implementation used by pyFAI's "
            "FiberIntegrator."
        )
        gi_form.addRow("Histogram backend", self.gi_method)
        self.gi_group.setVisible(False)
        layout.addWidget(self.gi_group)
        self.poni_v3_group = QtWidgets.QGroupBox(
            "Detector sensor / parallax (PONI v3)"
        )
        self.poni_v3_group.setObjectName("vnextPoniV3AdvancedGroup")
        poni_form = QtWidgets.QFormLayout(self.poni_v3_group)
        self.poni_v3_override = QtWidgets.QCheckBox(
            "Override selected PONI sensor settings"
        )
        self.poni_v3_override.setObjectName("vnextPoniV3Override")
        self.poni_v3_material = QtWidgets.QComboBox()
        self.poni_v3_material.setObjectName("vnextPoniV3Material")
        for material in PONI_V3_SENSOR_MATERIALS:
            self.poni_v3_material.addItem(material, material)
        self.poni_v3_thickness_m = QtWidgets.QLineEdit("0.00045")
        self.poni_v3_thickness_m.setObjectName("vnextPoniV3ThicknessM")
        self.poni_v3_thickness_m.setToolTip("450 µm = 0.00045 m")
        self.poni_v3_parallax = QtWidgets.QCheckBox("Enable parallax")
        self.poni_v3_parallax.setObjectName("vnextPoniV3Parallax")
        self.poni_v3_authority = QtWidgets.QLabel(
            "Override off: the selected PONI is authoritative. "
            "Override on: only sensor material, thickness, and parallax "
            "change for the next run."
        )
        self.poni_v3_authority.setWordWrap(True)
        poni_form.addRow(self.poni_v3_override)
        poni_form.addRow("Sensor material", self.poni_v3_material)
        poni_form.addRow("Sensor thickness (m)", self.poni_v3_thickness_m)
        poni_form.addRow(self.poni_v3_parallax)
        poni_form.addRow(self.poni_v3_authority)
        self.poni_v3_override.toggled.connect(
            self._set_poni_v3_enabled
        )
        self._set_poni_v3_enabled(False)
        layout.addWidget(self.poni_v3_group)
        forms = QtWidgets.QHBoxLayout()
        forms.setSpacing(12)
        self.one_d = self._dimension_group(
            forms,
            "Integrate 1D",
            ADVANCED_METHODS_1D,
        )
        self.two_d = self._dimension_group(
            forms,
            "Integrate 2D",
            ADVANCED_METHODS_2D,
        )
        layout.addLayout(forms)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(700, 590)

    def _set_poni_v3_enabled(self, enabled: bool) -> None:
        for widget in (
            self.poni_v3_material,
            self.poni_v3_thickness_m,
            self.poni_v3_parallax,
        ):
            widget.setEnabled(bool(enabled))

    def _dimension_group(
        self,
        parent_layout: QtWidgets.QHBoxLayout,
        title: str,
        methods: tuple[str, ...],
    ) -> _DimensionEditors:
        group = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(group)
        solid_angle = QtWidgets.QCheckBox("Correct solid angle")
        apply_polarization = QtWidgets.QCheckBox("Apply polarization")
        polarization_factor = QtWidgets.QDoubleSpinBox()
        polarization_factor.setDecimals(4)
        polarization_factor.setRange(-1.0, 1.0)
        polarization_factor.setSingleStep(0.01)
        method = QtWidgets.QComboBox()
        method.addItems(methods)
        dummy = QtWidgets.QLineEdit()
        delta_dummy = QtWidgets.QLineEdit()
        chi_offset = QtWidgets.QLineEdit()
        safe = QtWidgets.QCheckBox("Safe integration")

        form.addRow(solid_angle)
        form.addRow(apply_polarization)
        form.addRow("Polarization factor", polarization_factor)
        form.addRow("Method", method)
        form.addRow("Dummy", dummy)
        form.addRow("Dummy tolerance", delta_dummy)
        form.addRow("χ offset (°)", chi_offset)
        form.addRow(safe)
        apply_polarization.toggled.connect(
            polarization_factor.setEnabled
        )
        parent_layout.addWidget(group)
        return _DimensionEditors(
            solid_angle,
            apply_polarization,
            polarization_factor,
            method,
            dummy,
            delta_dummy,
            chi_offset,
            safe,
        )

    def edit(
        self,
        snapshot: RunIntentSnapshot,
    ) -> AdvancedSettingsValues | None:
        """Load one snapshot and return values only after explicit acceptance."""

        self.load_values(advanced_settings_values(snapshot))
        result = self.exec()
        if result != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return self.values()

    def load_values(self, values: AdvancedSettingsValues) -> None:
        if type(values) is not AdvancedSettingsValues:
            raise TypeError("values must be AdvancedSettingsValues")
        self._load_dimension(self.one_d, values.one_d)
        self._load_dimension(self.two_d, values.two_d)
        self._gi_enabled = values.gi_enabled
        gi_enabled = bool(values.gi_enabled)
        blocker = QtCore.QSignalBlocker(self.gi_method)
        method = str(values.gi_method)
        index = self.gi_method.findData(method)
        if index < 0:
            self.gi_method.addItem(method, method)
            index = self.gi_method.count() - 1
            self.gi_method.setItemData(
                index,
                "Current custom Fiber histogram backend.",
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        self.gi_method.setCurrentIndex(index)
        del blocker
        self.gi_group.setVisible(gi_enabled)
        self.gi_method.setEnabled(gi_enabled)
        blockers = tuple(
            QtCore.QSignalBlocker(widget)
            for widget in (
                self.poni_v3_override,
                self.poni_v3_material,
                self.poni_v3_thickness_m,
                self.poni_v3_parallax,
            )
        )
        self.poni_v3_override.setChecked(
            bool(values.poni_v3_override_enabled)
        )
        material = str(values.poni_v3_material)
        material_index = self.poni_v3_material.findData(material)
        self.poni_v3_material.setCurrentIndex(max(0, material_index))
        self.poni_v3_thickness_m.setText(str(values.poni_v3_thickness_m))
        self.poni_v3_parallax.setChecked(bool(values.poni_v3_parallax))
        del blockers
        self._set_poni_v3_enabled(
            bool(values.poni_v3_override_enabled)
        )
        standard_method_tooltip = (
            "This standard integration method is not used by "
            "FiberIntegrator in Grazing mode."
            if gi_enabled
            else ""
        )
        for editors in (self.one_d, self.two_d):
            editors.method.setEnabled(not gi_enabled)
            editors.method.setToolTip(standard_method_tooltip)

    @staticmethod
    def _load_dimension(
        editors: _DimensionEditors,
        values: AdvancedDimensionValues,
    ) -> None:
        blockers = tuple(
            QtCore.QSignalBlocker(widget)
            for widget in (
                editors.solid_angle,
                editors.apply_polarization,
                editors.polarization_factor,
                editors.method,
                editors.dummy,
                editors.delta_dummy,
                editors.chi_offset,
                editors.safe,
            )
        )
        editors.solid_angle.setChecked(bool(values.correct_solid_angle))
        editors.apply_polarization.setChecked(
            bool(values.apply_polarization)
        )
        editors.polarization_factor.setValue(
            float(values.polarization_factor)
        )
        method = str(values.method)
        index = editors.method.findText(method)
        if index < 0:
            editors.method.addItem(method)
            index = editors.method.count() - 1
            editors.method.setItemData(
                index,
                "Current custom integration method.",
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        editors.method.setCurrentIndex(index)
        editors.dummy.setText(_optional_text(values.dummy))
        editors.delta_dummy.setText(_optional_text(values.delta_dummy))
        editors.chi_offset.setText(str(values.chi_offset))
        editors.safe.setChecked(bool(values.safe))
        editors.polarization_factor.setEnabled(
            bool(values.apply_polarization)
        )
        del blockers

    def values(self) -> AdvancedSettingsValues:
        return AdvancedSettingsValues(
            one_d=self._dimension_values(self.one_d),
            two_d=self._dimension_values(self.two_d),
            gi_enabled=self._gi_enabled,
            gi_method=self.gi_method.currentData(),
            poni_v3_override_enabled=self.poni_v3_override.isChecked(),
            poni_v3_material=self.poni_v3_material.currentData(),
            poni_v3_thickness_m=self.poni_v3_thickness_m.text(),
            poni_v3_parallax=self.poni_v3_parallax.isChecked(),
        )

    @staticmethod
    def _dimension_values(
        editors: _DimensionEditors,
    ) -> AdvancedDimensionValues:
        return AdvancedDimensionValues(
            editors.solid_angle.isChecked(),
            editors.apply_polarization.isChecked(),
            editors.polarization_factor.value(),
            editors.method.currentText(),
            editors.dummy.text(),
            editors.delta_dummy.text(),
            editors.chi_offset.text(),
            editors.safe.isChecked(),
        )


def _optional_text(value: object) -> str:
    return "" if value is None else str(value)


__all__ = ["AdvancedSettingsDialog"]
