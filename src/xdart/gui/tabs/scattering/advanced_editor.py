"""Native value editor for revision-owned integration settings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

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
from xdart.gui.widgets.controls_panel import PillRow, SubsectionCard
from xdart.gui.widgets.file_dialogs import CORRECTION_FILTER as _CORRECTION_FILTER
from .controls_inventory import RECORD_ANGULAR_SUPPORT, recording_control_field

#: Browse titles for the detector-correction images, by role.
_CORRECTION_TITLES = {"dark": "Choose dark", "flat": "Choose flat field"}


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
    error_model: QtWidgets.QComboBox


class AdvancedDialog(QtWidgets.QDialog):
    """Int's Advanced dialog for any page: the rarely changed settings, in titled groups.

    The canonical owner of a page's Advanced settings (the architecture catalogue; Codex's reuse
    audits, 2026-10-01). A page builds each group from the shared Controls rows (``FormRow``,
    ``RangeRow``, the whole-number kind) and adds it with ``add_group``. Its rows commit as they are
    edited, so the dialog has no values of its own and closes with Close. Int's integration editor
    (``AdvancedSettingsDialog``) is the modal consumer: it adds its own groups with ``add_section``,
    and returns its values only on OK.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        title: str,
        object_name: str = "advancedDialog",
        accept_buttons: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setWindowTitle(title)
        self._accept_buttons = accept_buttons
        layout = QtWidgets.QVBoxLayout(self)
        self._sections = QtWidgets.QVBoxLayout()
        layout.addLayout(self._sections)
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            if accept_buttons
            else QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def add_group(self, title: str, rows=()) -> SubsectionCard:
        """A titled group of shared rows, below the groups before it; returns its card, to which
        more rows can be added."""
        card = SubsectionCard(title)
        for row in rows:
            card.add_row(row)
        self._sections.addWidget(card)
        if title == "Corrections" and not self._accept_buttons:
            from xdart.gui.widgets.context_help import owner_for
            card.set_help_topic("corrections.existing")
            self._help_owner = owner_for(self)
        return card

    def add_section(self, item: QtWidgets.QWidget | QtWidgets.QLayout) -> None:
        """An owner's own group (a widget or a layout), below the groups before it."""
        if isinstance(item, QtWidgets.QLayout):
            self._sections.addLayout(item)
        else:
            self._sections.addWidget(item)

    def set_editors_enabled(self, enabled: bool) -> None:
        """Lock nonmodal group bodies while leaving contextual header help available."""
        for index in range(self._sections.count()):
            card = self._sections.itemAt(index).widget()
            if isinstance(card, SubsectionCard):
                card.body.setEnabled(enabled)
        self.buttons.setEnabled(enabled)


class AdvancedSettingsDialog(AdvancedDialog):
    """A combined 1D/2D editor containing values, not legacy widgets."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        path_chooser: Callable[[str, str], str | None] | None = None,
        correction_inspector: Callable[[str, str, float], object] | None = None,
    ) -> None:
        super().__init__(
            parent,
            title="Integration — Advanced Settings",
            object_name="vnextAdvancedSettingsDialog",
            accept_buttons=True,
        )
        self.setModal(True)
        self._gi_enabled: object = False
        self._path_chooser = (
            path_chooser if path_chooser is not None else self._choose_file
        )
        # (role, path, flat cutoff) -> DetectorImageCheck, for the status lines.
        self._correction_inspector = correction_inspector
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
        self.add_section(self.gi_group)
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
        self.add_section(self.poni_v3_group)
        self.corrections_group = QtWidgets.QGroupBox("Detector corrections")
        self.corrections_group.setObjectName("vnextDetectorCorrectionsGroup")
        corrections_form = QtWidgets.QFormLayout(self.corrections_group)
        self.dark_file, self.dark_status = self._correction_row(
            corrections_form, "dark", "Dark",
            "Dark-current image subtracted from every frame before integration. "
            "With the Poisson error model its counts add to the variance. "
            "Background, if set, is subtracted as well, less its own dark.",
        )
        self.flat_file, self.flat_status = self._correction_row(
            corrections_form, "flat", "Flat field",
            "Flat-field image the integration divides by. Dead pixels and pixels "
            "below the cutoff are masked.",
        )
        self.flat_min_percent = QtWidgets.QDoubleSpinBox()
        self.flat_min_percent.setObjectName("vnextFlatMinPercent")
        self.flat_min_percent.setRange(0.0, 99.0)
        self.flat_min_percent.setDecimals(1)
        self.flat_min_percent.setSingleStep(1.0)
        self.flat_min_percent.setSuffix(" % of median")
        # Re-check on a committed value, not on every keystroke.
        self.flat_min_percent.setKeyboardTracking(False)
        self.flat_min_percent.setToolTip(
            "Flat pixels below this share of the flat's median are treated as dead "
            "and masked. Raise it to mask a region shadowed in the flat; lower it to "
            "keep weak but working pixels. 0 masks only zero, negative and "
            "non-finite pixels."
        )
        self.flat_min_percent.valueChanged.connect(
            lambda _value: self._refresh_status("flat")
        )
        corrections_form.addRow("Mask flat pixels below", self.flat_min_percent)
        self.add_section(self.corrections_group)
        forms = QtWidgets.QHBoxLayout()
        forms.setSpacing(12)
        self.one_d = self._dimension_group(
            forms,
            "Integrate 1D",
            ADVANCED_METHODS_1D,
        )
        self._recording_field = recording_control_field()
        self.record_angular_support = PillRow((self._recording_field,))
        self.one_d.solid_angle.parentWidget().layout().addRow(self.record_angular_support)
        self.two_d = self._dimension_group(
            forms,
            "Integrate 2D",
            ADVANCED_METHODS_2D,
        )
        self.add_section(forms)
        self.resize(700, 720)

    def _correction_row(
        self,
        form: QtWidgets.QFormLayout,
        role: str,
        label: str,
        tooltip: str,
    ) -> tuple[QtWidgets.QLineEdit, QtWidgets.QLabel]:
        path = QtWidgets.QLineEdit()
        path.setObjectName(f"vnext{role.title()}File")
        path.setPlaceholderText("None")
        path.setToolTip(tooltip)
        path.editingFinished.connect(lambda: self._refresh_status(role))
        browse = QtWidgets.QPushButton("Browse…")
        browse.setObjectName(f"vnext{role.title()}FileBrowse")
        browse.clicked.connect(lambda _checked=False: self._browse(role, path))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(path, 1)
        row.addWidget(browse)
        form.addRow(label, row)
        status = QtWidgets.QLabel()
        status.setObjectName(f"vnext{role.title()}FileStatus")
        status.setWordWrap(True)
        status.setVisible(False)
        form.addRow("", status)
        return path, status

    def _browse(self, role: str, editor: QtWidgets.QLineEdit) -> None:
        selected = self._path_chooser(role, editor.text())
        if type(selected) is str and selected:
            editor.setText(selected)
            self._refresh_status(role)

    def _refresh_status(self, role: str) -> None:
        """Check the dark or flat file and show what it would do, before a run."""
        editor, status = ((self.dark_file, self.dark_status) if role == "dark"
                          else (self.flat_file, self.flat_status))
        path = editor.text().strip()
        inspector = self._correction_inspector
        if not path or inspector is None:
            status.clear()
            status.setVisible(False)
            return
        try:
            check = inspector(role, path, self.flat_min_percent.value() / 100.0)
            text = ("Not usable: " if check.errors else "") + check.summary()
        except Exception as error:  # noqa: BLE001 - a failed check must not block editing
            text = f"Could not check the file: {error}"
        status.setText(text)
        status.setVisible(True)

    def _choose_file(self, role: str, current: str) -> str | None:
        from xdart.utils.browse import browse_start_dir, remember_browse_path

        selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, _CORRECTION_TITLES[role], browse_start_dir(current),
            _CORRECTION_FILTER,
        )
        if selected:
            remember_browse_path(selected)
        return selected or None

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
        error_model = QtWidgets.QComboBox()
        error_model.addItem("None", None)
        error_model.addItem("Poisson", "poisson")
        error_model.setToolTip(
            "Poisson propagates counting uncertainty and saves sigma with the "
            "result where pyFAI provides it: the q and 2θ profiles, and the "
            "grazing-incidence total-q profile over the full χ range. The χ "
            "profile, a χ-restricted total-q profile and the other "
            "grazing-incidence 1D profiles have none."
        )

        form.addRow(solid_angle)
        form.addRow(apply_polarization)
        form.addRow("Polarization factor", polarization_factor)
        form.addRow("Method", method)
        form.addRow("Dummy", dummy)
        form.addRow("Dummy tolerance", delta_dummy)
        form.addRow("χ offset (°)", chi_offset)
        form.addRow("Error model", error_model)
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
            error_model,
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
        self.record_angular_support.apply_fields({RECORD_ANGULAR_SUPPORT:
            replace(self._recording_field, value=values.record_angular_support)})
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
        self.dark_file.setText(str(values.dark_file))
        self.flat_file.setText(str(values.flat_file))
        blocker = QtCore.QSignalBlocker(self.flat_min_percent)
        self.flat_min_percent.setValue(100.0 * float(values.flat_min_fraction))
        del blocker
        for role in ("dark", "flat"):
            self._refresh_status(role)
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
                editors.error_model,
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
        model = values.error_model
        index = editors.error_model.findData(model)
        if index < 0:
            editors.error_model.addItem(str(model), model)
            index = editors.error_model.count() - 1
            editors.error_model.setItemData(
                index,
                "Current custom error model.",
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        editors.error_model.setCurrentIndex(index)
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
            dark_file=self.dark_file.text(),
            flat_file=self.flat_file.text(),
            flat_min_fraction=round(self.flat_min_percent.value(), 1) / 100.0,
            record_angular_support=dict(self.record_angular_support.current_edits())[RECORD_ANGULAR_SUPPORT],
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
            editors.error_model.currentData(),
        )


def _optional_text(value: object) -> str:
    return "" if value is None else str(value)


__all__ = ["AdvancedDialog", "AdvancedSettingsDialog"]
