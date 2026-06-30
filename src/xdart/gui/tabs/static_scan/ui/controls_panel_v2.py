# -*- coding: utf-8 -*-
"""Controls Panel V2 scaffold.

This widget is a thin renderer for the Qt-free
:mod:`xdart.gui.tabs.static_scan.controls_logic` profile.  During the
ParameterTree migration it also renders a small set of bound form rows supplied
by the owning tab; those rows emit field-change intent and do not read wrangler
objects directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pyqtgraph.Qt import QtCore, QtWidgets

from ..controls_logic import (
    AnalysisLauncherSpec,
    BoundControlState,
    ControlAction,
    ControlActionSpec,
    ControlFieldKind,
    ControlFormEdit,
    ControlFormField,
    ControlPanelRenderState,
    ControlProfile,
    FieldId,
    FieldStatus,
    SectionId,
    build_bound_control_state,
)

# Order: Project · Experiment · Source · Processing (instrument config before
# data, matching the notebook workflow, Vivek).  The number follows the order;
# the accent follows the section identity.
_SECTION_META = {
    SectionId.PROJECT: ("1", "PROJECT", "project"),
    SectionId.EXPERIMENT: ("2", "EXPERIMENT", "experiment"),
    SectionId.SOURCE: ("3", "SOURCE", "source"),
    SectionId.PROCESSING: ("4", "PROCESSING", "processing"),
    SectionId.OUTPUT: ("", "OUTPUT", "neutral"),
    SectionId.ANALYSIS: ("", "ANALYSIS", "neutral"),
}

_ACTION_LABELS = {
    ControlAction.CALIBRATE: "⌖ Calibrate",
    ControlAction.REFINE_GEOMETRY: "◎ Refine",
    ControlAction.MAKE_MASK: "▦ Make Mask",
}

# Descriptive hover tooltips, keyed by bound-control path.  Used by every row
# renderer (FormRow/PillRow/SegmentedControl); a field's disabled `reason` is
# used as a fallback when there is no entry here.
_FIELD_TOOLTIPS: dict[tuple[str, ...], str] = {
    # Experiment / grazing-incidence
    ("GI", "Grazing"): (
        "Measurement geometry: Standard, or Grazing-incidence (GIWAXS), which "
        "enables the fiber/GI integration axes and corrections."),
    ("GI", "th_motor"): (
        "Incidence-angle source: the scan-metadata motor column read as the "
        "grazing incidence angle θ. Pick 'Manual' to enter θ directly."),
    ("GI", "th_val"): (
        "Manual grazing incidence angle θ in degrees (used when the motor is "
        "set to 'Manual')."),
    ("GI", "sample_orientation"): (
        "Sample orientation index (1–8): the pyFAI fiber sample-frame "
        "convention for the grazing geometry."),
    ("GI", "tilt_angle"): "Sample tilt angle in degrees for the grazing geometry.",
    # Processing / conditioning
    ("MaskSat", "mask_sentinel"): (
        "Mask dead/saturated detector pixels (the uint32 sentinel). OFF keeps "
        "strong saturated Bragg peaks unmasked."),
    ("Signal", "series_average"): (
        "Average all frames in the series into one frame before integration."),
    ("Mask", "Threshold"): (
        "Clip pixel intensities outside the [min, max] band before integration."),
    # Source
    ("Signal", "inp_type"): (
        "Source kind: a numbered image series, a directory of images, or a "
        "single image."),
    ("Signal", "include_subdir"): "Also search sub-directories for matching images.",
    ("Signal", "img_ext"): "Image file type (extension).",
    ("Signal", "meta_ext"): "Per-frame metadata format (e.g. txt, SPEC, pdi).",
    ("Signal", "meta_dir"): (
        "Directory holding the SPEC metadata file, when it's separate from the "
        "images."),
}

# Descriptive hover tooltips for the producer/inspector action buttons.
_ACTION_TOOLTIPS: dict[ControlAction, str] = {
    ControlAction.CALIBRATE: "Run pyFAI calibration to produce a PONI (detector geometry) file.",
    ControlAction.MAKE_MASK: "Build a detector mask from the current frame.",
    ControlAction.REFINE_GEOMETRY: "Refine the diffractometer geometry from the loaded scan.",
    ControlAction.ADVANCED_PROCESSING: "Open the advanced integration settings (full parameter tree).",
    ControlAction.REINTEGRATE_1D: "Re-integrate the loaded scan to 1D with the current settings.",
    ControlAction.REINTEGRATE_2D: "Re-integrate the loaded scan to 2D with the current settings.",
}


def _field_tooltip(path, reason: str = "") -> str:
    """Resolve a control's hover tooltip: the disabled `reason` takes precedence
    (so a locked control explains WHY), else the descriptive help text."""
    return (reason or "") or _FIELD_TOOLTIPS.get(tuple(path), "")


class StatusBadge(QtWidgets.QLabel):
    """Small status label used by card rows."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setObjectName("controlsV2StatusBadge")

    def set_status(self, text: str, severity: str = "info") -> None:
        self.setText(text)
        self.setProperty("severity", severity)
        self.style().unpolish(self)
        self.style().polish(self)


class LauncherButton(QtWidgets.QPushButton):
    """Analysis launcher button carrying its launcher spec."""

    launched = QtCore.Signal(object)

    def __init__(self, spec: AnalysisLauncherSpec, parent=None):
        super().__init__(spec.label, parent)
        self.setObjectName("controlsV2LauncherButton")
        self._spec = spec
        self.clicked.connect(lambda: self.launched.emit(self._spec.tool))
        self.apply_spec(spec)

    @property
    def spec(self) -> AnalysisLauncherSpec:
        return self._spec

    def apply_spec(self, spec: AnalysisLauncherSpec) -> None:
        self._spec = spec
        self.setText(spec.label)
        self.setEnabled(bool(spec.enabled))
        self.setToolTip(spec.reason or "")
        self.setProperty("productionReady", bool(spec.production_ready))
        self.style().unpolish(self)
        self.style().polish(self)


class ActionButton(QtWidgets.QPushButton):
    """Small card action button carrying a ``ControlActionSpec``."""

    actionRequested = QtCore.Signal(object)

    def __init__(self, spec: ControlActionSpec, parent=None):
        super().__init__(spec.label, parent)
        self.setObjectName("controlsV2ActionButton")
        self._spec = spec
        self.clicked.connect(lambda: self.actionRequested.emit(self._spec.action))
        self.apply_spec(spec)

    @property
    def spec(self) -> ControlActionSpec:
        return self._spec

    def apply_spec(self, spec: ControlActionSpec) -> None:
        self._spec = spec
        self.setText(spec.label)
        self.setEnabled(bool(spec.enabled))
        self.setToolTip(spec.reason or _ACTION_TOOLTIPS.get(spec.action, ""))
        self.setProperty("productionReady", bool(spec.production_ready))
        # Role drives a subtle tint: Reintegrate = green (run-like), Advanced =
        # red (the destructive/expert escape hatch); producers stay neutral.
        if spec.action in (ControlAction.REINTEGRATE_1D, ControlAction.REINTEGRATE_2D):
            role = "reintegrate"
        elif spec.action == ControlAction.ADVANCED_PROCESSING:
            role = "advanced"
        else:
            role = ""
        self.setProperty("actionRole", role)
        self.style().unpolish(self)
        self.style().polish(self)


class SectionCard(QtWidgets.QFrame):
    """Colour-coded collapsible workflow section."""

    def __init__(
        self,
        title: str,
        parent=None,
        *,
        number: str = "",
        accent: str = "neutral",
        collapsible: bool = True,
    ):
        super().__init__(parent)
        self.setObjectName("controlsV2SectionCard")
        self.setProperty("accent", accent)
        self._collapsed = False
        self._collapsible = bool(collapsible)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.header = QtWidgets.QFrame()
        self.header.setObjectName("controlsV2SectionHeader")
        self.header.setProperty("accent", accent)
        header_lay = QtWidgets.QHBoxLayout(self.header)
        header_lay.setContentsMargins(7, 5, 8, 5)
        header_lay.setSpacing(7)
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setObjectName("controlsV2Chevron")
        self.toggle.setText("▾")
        self.toggle.setAutoRaise(True)
        self.toggle.setToolTip("Collapse section")
        self.toggle.setEnabled(self._collapsible)
        self.toggle.clicked.connect(self.toggle_collapsed)
        header_lay.addWidget(self.toggle)
        self.chip = QtWidgets.QLabel(number)
        self.chip.setObjectName("controlsV2SectionChip")
        self.chip.setProperty("accent", accent)
        self.chip.setAlignment(QtCore.Qt.AlignCenter)
        self.chip.setVisible(bool(number))
        header_lay.addWidget(self.chip)
        self.title = QtWidgets.QLabel(title.upper())
        self.title.setObjectName("controlsV2SectionTitle")
        header_lay.addWidget(self.title, 1)
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("controlsV2SectionStatus")
        self.status.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.status.hide()
        header_lay.addWidget(self.status)
        self.header.mousePressEvent = self._header_mouse_press
        outer.addWidget(self.header)

        self.body = QtWidgets.QFrame()
        self.body.setObjectName("controlsV2SectionBody")
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(8, 7, 8, 8)
        self.body_layout.setSpacing(5)
        outer.addWidget(self.body)

        self.embedded = QtWidgets.QFrame()
        self.embedded.setObjectName("controlsV2Embedded")
        self.embedded_layout = QtWidgets.QVBoxLayout(self.embedded)
        self.embedded_layout.setContentsMargins(0, 4, 0, 0)
        self.embedded_layout.setSpacing(0)
        self.embedded.hide()
        self.body_layout.addWidget(self.embedded)

    def toggle_collapsed(self) -> None:
        if self._collapsible:
            self.set_collapsed(not self._collapsed)

    def _header_mouse_press(self, event) -> None:
        if (
            self._collapsible
            and event.button() == QtCore.Qt.LeftButton
        ):
            self.toggle_collapsed()
            event.accept()
            return
        QtWidgets.QFrame.mousePressEvent(self.header, event)

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = bool(collapsed)
        self.body.setVisible(not self._collapsed)
        self.toggle.setText("▸" if self._collapsed else "▾")
        self.toggle.setToolTip("Expand section" if self._collapsed else "Collapse section")

    def set_status_text(self, text: str = "") -> None:
        self.status.setText(text)
        self.status.setVisible(bool(text))

    def clear_rows(self) -> None:
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                if widget is self.embedded:
                    continue
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        if self.embedded.parent() is not self.body:
            self.embedded.setParent(self.body)
        self.body_layout.addWidget(self.embedded)

    def add_row(self, widget: QtWidgets.QWidget) -> None:
        self.body_layout.addWidget(widget)

    def add_group_header(self, text: str) -> None:
        label = QtWidgets.QLabel(text)
        label.setObjectName("controlsV2GroupHeader")
        self.add_row(label)

    def set_embedded_widget(
        self,
        widget: QtWidgets.QWidget | None,
        *,
        visible: bool = True,
    ) -> None:
        current = self.embedded_layout.itemAt(0)
        if current is not None and current.widget() is widget:
            self.embedded.setVisible(widget is not None and visible)
            if widget is not None:
                widget.setVisible(visible)
            return
        while self.embedded_layout.count():
            item = self.embedded_layout.takeAt(0)
            old = item.widget()
            if old is not None:
                old.setParent(None)
        if widget is None:
            self.embedded.hide()
            return
        self.embedded_layout.addWidget(widget)
        widget.setVisible(visible)
        self.embedded.setVisible(visible)


class SubsectionCard(QtWidgets.QFrame):
    """Compact collapsible subsection inside a workflow section."""

    def __init__(
        self,
        title: str,
        parent=None,
        *,
        prefix: str = "",
        status: str = "",
        accent: str = "neutral",
    ):
        super().__init__(parent)
        self.setObjectName("controlsV2SubsectionCard")
        self._collapsed = False
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.header = QtWidgets.QFrame()
        self.header.setObjectName("controlsV2SubsectionHeader")
        h_lay = QtWidgets.QHBoxLayout(self.header)
        h_lay.setContentsMargins(7, 4, 7, 4)
        h_lay.setSpacing(6)
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setObjectName("controlsV2SubChevron")
        self.toggle.setText("▾")
        self.toggle.setAutoRaise(True)
        self.toggle.clicked.connect(self.toggle_collapsed)
        h_lay.addWidget(self.toggle)
        if prefix:
            self.prefix = QtWidgets.QLabel(prefix)
            self.prefix.setObjectName("controlsV2SubsectionPrefix")
            # The prefix chip takes the parent section's accent (amber for
            # Experiment, violet for Processing, …), not the global purple.
            self.prefix.setProperty("accent", accent)
            h_lay.addWidget(self.prefix)
        else:
            self.prefix = None
        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("controlsV2SubsectionTitle")
        # The title carries the section accent (amber for Experiment, red for
        # Processing, …) so every subsection reads consistently — no number chip.
        self.title.setProperty("accent", accent)
        # When a numbered/chip prefix already names the group (e.g. the "1-D"
        # chip), an identical title text duplicates it ("1-D 1-D").  Hide the
        # title label when empty so the chip alone names the group.
        self.title.setVisible(bool(title))
        h_lay.addWidget(self.title)
        # Explicit stretch (not the title's) so trailing header widgets (the
        # "Pts" cluster, the status) right-align even when the title is hidden —
        # a hidden title item would otherwise swallow its own stretch.
        h_lay.addStretch(1)
        # Trailing header slot for compact controls that belong on the header row
        # (e.g. the "Pts" field(s) for 1-D / 2-D), mirroring the mockup.
        self.header_extra = QtWidgets.QWidget()
        self.header_extra.setObjectName("controlsV2SubsectionHeaderExtra")
        self.header_extra_layout = QtWidgets.QHBoxLayout(self.header_extra)
        self.header_extra_layout.setContentsMargins(0, 0, 0, 0)
        self.header_extra_layout.setSpacing(5)
        self.header_extra.hide()
        h_lay.addWidget(self.header_extra)
        self.status = QtWidgets.QLabel(status)
        self.status.setObjectName("controlsV2SubsectionStatus")
        self.status.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.status.setVisible(bool(status))
        h_lay.addWidget(self.status)
        self.header.mousePressEvent = self._header_mouse_press
        outer.addWidget(self.header)
        self.body = QtWidgets.QFrame()
        self.body.setObjectName("controlsV2SubsectionBody")
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(7, 5, 7, 7)
        self.body_layout.setSpacing(4)
        outer.addWidget(self.body)

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def _header_mouse_press(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.toggle_collapsed()
            event.accept()
            return
        QtWidgets.QFrame.mousePressEvent(self.header, event)

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = bool(collapsed)
        self.body.setVisible(not self._collapsed)
        self.toggle.setText("▸" if self._collapsed else "▾")

    def add_row(self, widget: QtWidgets.QWidget) -> None:
        self.body_layout.addWidget(widget)

    def add_header_widget(self, widget: QtWidgets.QWidget) -> None:
        self.header_extra_layout.addWidget(widget)
        self.header_extra.show()


class FieldRow(QtWidgets.QWidget):
    """One typed field row rendered from ``FieldStatus``."""

    def __init__(self, status: FieldStatus, parent=None):
        super().__init__(parent)
        self.setObjectName("controlsV2FieldRow")
        self._status = status
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)
        self.label = QtWidgets.QLabel(status.label)
        self.label.setObjectName("controlsV2FieldLabel")
        self.label.setMinimumWidth(72)
        self.value = QtWidgets.QLabel(status.value or status.reason or status.status.value)
        self.value.setObjectName("controlsV2FieldValue")
        self.value.setWordWrap(False)
        self.value.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.badge = StatusBadge(status.status.value)
        self.badge.set_status(status.status.value, status.status.value)
        self.badge.setMinimumWidth(48)
        self.badge.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lay.addWidget(self.label)
        lay.addWidget(self.value, 1)
        lay.addWidget(self.badge)
        self.setToolTip(status.reason or status.headless_key or status.session_key)

    @property
    def status(self) -> FieldStatus:
        return self._status


class FormRow(QtWidgets.QWidget):
    """One editable row in the transitional V2 form."""

    valueChanged = QtCore.Signal(object, object)
    browseRequested = QtCore.Signal(object)

    def __init__(
        self,
        *,
        label: str,
        path: tuple[str, ...],
        value,
        kind: str = "line",
        choices: Sequence[str] = (),
        browse: bool = False,
        enabled: bool = True,
        reason: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("controlsV2FormRow")
        self._path = tuple(path)
        self._kind = kind
        self._enabled = bool(enabled)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.label = QtWidgets.QLabel(label)
        self.label.setObjectName("controlsV2FieldLabel")
        self.label.setMinimumWidth(76)

        if kind == "bool":
            # The legacy integrator panel uses checkable QPushButtons for
            # GI/threshold/saturation toggles.  Mirroring that behavior keeps
            # checked-but-disabled controls visibly checked during active runs.
            self.label.hide()
            editor = QtWidgets.QPushButton(label)
            editor.setObjectName("controlsV2ToggleButton")
            editor.setCheckable(True)
            editor.setChecked(bool(value))
            editor.toggled.connect(
                lambda checked: self._emit_edit(bool(checked))
            )
            editor.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Preferred,
            )
            lay.addWidget(editor, 1)
        elif kind == "combo":
            lay.addWidget(self.label)
            editor = QtWidgets.QComboBox()
            editor.setObjectName("controlsV2ComboBox")
            values = [str(choice) for choice in choices]
            if not values and value not in (None, ""):
                values = [str(value)]
            editor.addItems(values)
            text = "" if value is None else str(value)
            idx = editor.findText(text)
            if idx >= 0:
                editor.setCurrentIndex(idx)
            elif text:
                editor.addItem(text)
                editor.setCurrentText(text)
            editor.currentTextChanged.connect(
                lambda text: self._emit_edit(text)
            )
            lay.addWidget(editor, 1)
        else:
            lay.addWidget(self.label)
            editor = QtWidgets.QLineEdit("" if value is None else str(value))
            editor.setObjectName("controlsV2LineEdit")
            editor.editingFinished.connect(
                lambda e=editor: self._emit_edit(e.text())
            )
            lay.addWidget(editor, 1)

        self.editor = editor
        self.editor.setEnabled(self._enabled)
        # Path/browse fields should show the root of long paths by default; the
        # full value is still available on hover via the row tooltip.
        if browse and isinstance(self.editor, QtWidgets.QLineEdit):
            self.editor.setCursorPosition(0)
        self.browse_button = None
        if browse:
            btn = QtWidgets.QToolButton()
            btn.setText("📁")
            btn.setObjectName("controlsV2BrowseButton")
            btn.setToolTip(f"Browse {label}")
            btn.clicked.connect(lambda _=False, p=self._path: self.browseRequested.emit(p))
            btn.setMinimumWidth(31)
            btn.setMaximumWidth(36)
            btn.setEnabled(self._enabled)
            lay.addWidget(btn)
            self.browse_button = btn
        self.setToolTip(reason or "")

    @property
    def path(self) -> tuple[str, ...]:
        return self._path

    def add_trailing_widget(self, widget: QtWidgets.QWidget) -> None:
        """Append a widget to the right of the row (after the editor).  Used to
        park the compact 'Pts' cluster on the Axis row — the editor stretches, so
        trailing widgets pin to the right."""
        self.layout().addWidget(widget)

    def current_value(self):
        if self._kind == "bool":
            return bool(self.editor.isChecked())
        if self._kind == "combo":
            return self.editor.currentText()
        return self.editor.text()

    def _emit_edit(self, value) -> None:
        edit = ControlFormEdit(path=self._path, value=value)
        self.valueChanged.emit(edit.path, edit.value)


class RangeRow(QtWidgets.QWidget):
    """A compact range row: ``[label] [low] – [high] [✦]`` on one line.

    Coalesces a low/high pair (and an optional auto/enable toggle) that the
    keystone still emits as separate fields, so the panel matches the mockup
    without changing the field model or the write-through: each editor emits
    ``valueChanged(path, value)`` for the SAME path the individual rows used.
    """

    valueChanged = QtCore.Signal(object, object)

    def __init__(self, *, label, low, high, toggle=None, parent=None):
        super().__init__(parent)
        self.setObjectName("controlsV2RangeRow")
        self._entries: list[tuple[tuple[str, ...], str]] = []
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)

        self.label = QtWidgets.QLabel(label)
        self.label.setObjectName("controlsV2FieldLabel")
        self.label.setMinimumWidth(72)
        lay.addWidget(self.label)

        self._low, self._low_path = self._edit(low, lay)
        dash = QtWidgets.QLabel("–")
        dash.setObjectName("controlsV2RangeDash")
        dash.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(dash)
        self._high, self._high_path = self._edit(high, lay)

        self._toggle = None
        if toggle is not None:
            btn = QtWidgets.QToolButton()
            btn.setObjectName("controlsV2AutoButton")
            btn.setText("✦")
            btn.setCheckable(True)
            btn.setChecked(bool(toggle.get("value")))
            btn.setEnabled(bool(toggle.get("enabled", True)))
            btn.setToolTip(toggle.get("tooltip", "Auto"))
            btn.setMinimumWidth(31)
            tpath = tuple(toggle["path"])
            btn.toggled.connect(lambda checked, p=tpath: self.valueChanged.emit(p, bool(checked)))
            lay.addWidget(btn)
            self._toggle = (tpath, btn)

    def _edit(self, spec, lay):
        value = spec.get("value")
        edit = QtWidgets.QLineEdit("" if value is None else str(value))
        edit.setObjectName("controlsV2LineEdit")
        edit.setEnabled(bool(spec.get("enabled", True)))
        path = tuple(spec["path"])
        edit.editingFinished.connect(
            lambda e=edit, p=path: self.valueChanged.emit(p, e.text())
        )
        lay.addWidget(edit, 1)
        return edit, path

    def current_edits(self) -> tuple[tuple[tuple[str, ...], object], ...]:
        out = [
            (tuple(self._low_path), self._low.text()),
            (tuple(self._high_path), self._high.text()),
        ]
        if self._toggle is not None:
            out.append((self._toggle[0], bool(self._toggle[1].isChecked())))
        return tuple(out)


class PillRow(QtWidgets.QWidget):
    """A left-aligned row of compact toggle *pills* (the bool toggles).

    Replaces the full-width stacked toggle buttons with mockup-style pills:
    several share a line, each sized to its label.  Each pill emits
    ``valueChanged(path, bool)`` so the write-through is unchanged.
    """

    valueChanged = QtCore.Signal(object, object)

    def __init__(self, fields: Sequence[ControlFormField], parent=None):
        super().__init__(parent)
        self.setObjectName("controlsV2PillRow")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._pills: list[tuple[tuple[str, ...], QtWidgets.QPushButton]] = []
        for field in fields:
            btn = QtWidgets.QPushButton(field.label)
            # Reuse the accent-when-checked toggle styling, but content-sized.
            btn.setObjectName("controlsV2ToggleButton")
            btn.setProperty("pill", True)
            btn.setCheckable(True)
            btn.setChecked(bool(field.value))
            btn.setEnabled(bool(field.enabled))
            btn.setToolTip(_field_tooltip(field.path, field.reason))
            btn.setSizePolicy(
                QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed
            )
            path = tuple(field.path)
            btn.toggled.connect(
                lambda checked, p=path: self.valueChanged.emit(p, bool(checked))
            )
            lay.addWidget(btn)
            self._pills.append((path, btn))
        lay.addStretch(1)

    def current_edits(self) -> tuple[tuple[tuple[str, ...], object], ...]:
        return tuple((p, bool(btn.isChecked())) for p, btn in self._pills)


class SegmentedControl(QtWidgets.QWidget):
    """A mutually-exclusive segmented toggle (e.g. ``Standard | Grazing``).

    Replaces a group-header bool checkbox with a proper two-state segmented
    control — the header checkbox was the #56 repaint class.  Built as an
    exclusive ``QButtonGroup`` of checkable buttons; ``options`` maps each segment
    label to the value emitted when it becomes the active selection.  It emits
    ``valueChanged(path, value)`` for the SAME path the old single bool toggle
    used, so the write-through is unchanged.  Reusable: Stitch GI drives the same
    ``("GI", "Grazing")`` path through this control.
    """

    valueChanged = QtCore.Signal(object, object)

    def __init__(self, path, options, *, value, enabled=True, reason="",
                 parent=None):
        super().__init__(parent)
        self.setObjectName("controlsV2SegmentedControl")
        self._path = tuple(path)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        self._segments: list[tuple[object, QtWidgets.QPushButton]] = []
        for idx, (label, opt_value) in enumerate(options):
            btn = QtWidgets.QPushButton(label)
            btn.setObjectName("controlsV2SegmentButton")
            btn.setProperty("segment", True)
            btn.setCheckable(True)
            btn.setEnabled(bool(enabled))
            btn.setToolTip(_field_tooltip(self._path, reason))
            btn.setChecked(opt_value == value)
            btn.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
            )
            self._group.addButton(btn, idx)
            # Use `clicked` (not `toggled`): an exclusive group fires `toggled` on
            # BOTH the newly-checked and newly-unchecked buttons; `clicked` fires
            # once, on the button the user pressed.  Emit the segment's fixed
            # value so the write-through gets (path, True/False) regardless of the
            # prior selection (re-clicking the active segment re-emits the same
            # value — the equality-guarded write-through makes that a no-op).
            btn.clicked.connect(
                lambda _checked, v=opt_value: self.valueChanged.emit(self._path, v)
            )
            lay.addWidget(btn, 1)
            self._segments.append((opt_value, btn))

    @property
    def path(self) -> tuple[str, ...]:
        return self._path

    def current_value(self):
        for opt_value, btn in self._segments:
            if btn.isChecked():
                return opt_value
        return None

    def current_edits(self) -> tuple[tuple[tuple[str, ...], object], ...]:
        value = self.current_value()
        if value is None:
            return ()
        return ((self._path, value),)


class ControlsPanelV2(QtWidgets.QWidget):
    """Feature-flag-ready renderer for :class:`ControlProfile`.

    It emits launcher intent only.  The owning tab decides how to open dialogs,
    run scans, or map profile changes into the legacy wrangler while V2 is
    hidden.
    """

    analysisLaunchRequested = QtCore.Signal(object)
    controlActionRequested = QtCore.Signal(object)
    fieldValueChanged = QtCore.Signal(object, object)
    fieldBrowseRequested = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("controlsPanelV2")
        self.setMinimumWidth(360)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(5, 5, 5, 5)
        # Roomier gap between the workflow sections for visual separation (Vivek).
        lay.setSpacing(12)

        self.top_action_bar = QtWidgets.QWidget()
        self.top_action_bar.setObjectName("controlsV2TopActionBar")
        self.top_action_layout = QtWidgets.QHBoxLayout(self.top_action_bar)
        self.top_action_layout.setContentsMargins(0, 0, 0, 0)
        self.top_action_layout.setSpacing(7)
        self.top_action_bar.hide()

        self.summary_card = SectionCard("Run Readiness", collapsible=False)
        self.project_card = self._make_section(SectionId.PROJECT)
        self.source_card = self._make_section(SectionId.SOURCE)
        self.experiment_card = self._make_section(SectionId.EXPERIMENT)
        self.processing_card = self._make_section(SectionId.PROCESSING)
        self.output_card = self._make_section(SectionId.OUTPUT)
        self.analysis_card = self._make_section(SectionId.ANALYSIS)
        lay.addWidget(self.top_action_bar)
        lay.addWidget(self.summary_card)
        lay.addWidget(self.project_card)
        lay.addWidget(self.experiment_card)
        lay.addWidget(self.source_card)
        lay.addWidget(self.processing_card)
        lay.addWidget(self.output_card)
        lay.addWidget(self.analysis_card)
        lay.addStretch(1)
        self._profile = None
        self._bound_state: BoundControlState | None = None

    @staticmethod
    def _make_section(section: SectionId) -> SectionCard:
        number, title, accent = _SECTION_META[section]
        return SectionCard(title, number=number, accent=accent)

    def set_processing_widget(
        self,
        widget: QtWidgets.QWidget | None,
        *,
        visible: bool = True,
    ) -> None:
        self.processing_card.set_embedded_widget(widget, visible=visible)

    @property
    def profile(self) -> ControlProfile | None:
        return self._profile

    def set_field_values(
        self,
        values: dict[tuple[str, ...], object] | None = None,
        choices: dict[tuple[str, ...], Sequence[str]] | None = None,
    ) -> None:
        self.set_bound_state(build_bound_control_state(values, choices))

    def set_bound_state(self, state: BoundControlState | None) -> None:
        self._bound_state = state
        if self._profile is not None:
            self._render_fields(self._profile)
            self._render_analysis(self._profile.analysis_launchers)

    def set_state(self, state: ControlPanelRenderState) -> None:
        """Render one immutable Controls V2 state snapshot."""
        self._bound_state = state.bound_controls
        self.set_profile(state.profile)

    def set_profile(self, profile: ControlProfile) -> None:
        self._profile = profile
        self._render_summary(profile)
        self._render_fields(profile)
        self._render_analysis(profile.analysis_launchers)

    def current_form_edits(self) -> tuple[ControlFormEdit, ...]:
        """Return the current visible editor values, including focused line edits."""

        edits = [
            ControlFormEdit(path=row.path, value=row.current_value())
            for row in self.findChildren(FormRow)
        ]
        for row in self.findChildren(RangeRow):
            edits.extend(
                ControlFormEdit(path=p, value=v) for p, v in row.current_edits()
            )
        for row in self.findChildren(SegmentedControl):
            edits.extend(
                ControlFormEdit(path=p, value=v) for p, v in row.current_edits()
            )
        return tuple(edits)

    def _render_summary(self, profile: ControlProfile) -> None:
        self.summary_card.clear_rows()
        if profile.run_enabled:
            badge = StatusBadge("Ready")
            badge.set_status("Ready", "ok")
            self.summary_card.add_row(badge)
            return
        blockers = profile.run_blockers or ("No run action in this mode.",)
        for blocker in blockers:
            badge = StatusBadge(blocker)
            badge.set_status(blocker, "blocked")
            self.summary_card.add_row(badge)

    def _render_fields(self, profile: ControlProfile) -> None:
        self._render_top_actions(profile)
        if self._bound_state is not None:
            self._render_bound_fields(profile)
            return
        sections = (
            (self.project_card, SectionId.PROJECT),
            (self.source_card, SectionId.SOURCE),
            (self.experiment_card, SectionId.EXPERIMENT),
            (self.processing_card, SectionId.PROCESSING),
            (self.output_card, SectionId.OUTPUT),
        )
        for card, section in sections:
            card.clear_rows()
            for status in profile.fields_for(section):
                card.add_row(FieldRow(status))
            if section != SectionId.EXPERIMENT:
                self._add_actions(card, profile.actions_for(section))
        self.experiment_card.setVisible(bool(profile.show_experiment_card))
        self.processing_card.setVisible(bool(profile.show_processing_card))

    def _render_bound_fields(self, profile: ControlProfile) -> None:
        # Tear down any open GI '…' popup on every rebuild.  Its rows are parented
        # under this panel, so current_form_edits() (findChildren(FormRow)) would
        # otherwise harvest a STALE popup whose displayed value froze at open time
        # — a later _commit_controls_v2_pending_edits could then clobber a fresher
        # sample_orientation/tilt_angle back to the old value (F1).  Leaving
        # Grazing triggers a rebuild too, so this also disposes the orphan popup
        # (F2).  Popup edits already write through on change, so nothing is lost;
        # reopening rebuilds its rows from the live profile.
        self._close_gi_more_popup()
        cards = {
            SectionId.PROJECT: self.project_card,
            SectionId.SOURCE: self.source_card,
            SectionId.EXPERIMENT: self.experiment_card,
            SectionId.PROCESSING: self.processing_card,
            SectionId.OUTPUT: self.output_card,
        }
        for card in cards.values():
            card.clear_rows()
            card.set_status_text("")

        state = self._bound_state or BoundControlState()
        self.summary_card.hide()
        self._render_plain_bound_section(
            self.project_card, state.fields_for(SectionId.PROJECT))
        self._render_source_bound_section(
            state.fields_for(SectionId.SOURCE))
        self._render_experiment_bound_section(
            profile, state.fields_for(SectionId.EXPERIMENT))
        self._render_processing_bound_section(
            profile, state.fields_for(SectionId.PROCESSING))

        self.source_card.set_status_text(self._source_status(profile))
        self.experiment_card.set_status_text(self._experiment_status(
            state.fields_for(SectionId.EXPERIMENT)))
        self.processing_card.set_status_text(self._processing_status(profile))
        self.experiment_card.setVisible(bool(state.fields_for(SectionId.EXPERIMENT)))
        self.processing_card.setVisible(True)
        self.output_card.setVisible(False)
        self.analysis_card.setVisible(False)

    def _render_plain_bound_section(
        self,
        card: SectionCard,
        fields: Iterable[ControlFormField],
    ) -> None:
        for field in fields:
            self._add_bound_row(card, field)

    def _render_source_bound_section(
        self,
        fields: Iterable[ControlFormField],
    ) -> None:
        """SOURCE layout with a few coalesced rows (Vivek): the mode combo + the
        directory-only Subdirs toggle share a row, and File Type + Meta Type share
        a row.  Everything else renders one field per row, in field order."""
        fields = tuple(fields)
        by_path = {field.path: field for field in fields}
        rendered: set[tuple[str, ...]] = set()

        def row_for(*paths: tuple[str, ...], stretches=None, tight=()) -> None:
            present = [by_path[p] for p in paths
                       if p in by_path and p not in rendered]
            if not present:
                return
            rendered.update(field.path for field in present)
            if len(present) == 1 and not tight:
                self._add_bound_row(self.source_card, present[0])
                return
            container = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(container)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(8)
            for field in present:
                sub = self._make_bound_row(field)
                if field.path in tight:
                    # Tighten the label against its editor (less gap); the editor
                    # still fills to the right edge of the cell (right-justified).
                    if getattr(sub, "label", None) is not None:
                        sub.label.setMinimumWidth(0)
                    if sub.layout() is not None:
                        sub.layout().setSpacing(3)
                is_bool = getattr(field.kind, "value", field.kind) == "bool"
                st = 0 if is_bool else 1
                if stretches and field.path in stretches:
                    st = stretches[field.path]
                lay.addWidget(sub, st)
            self.source_card.add_row(container)

        # Mode combo + (directory-only) Subdirs toggle on one row.
        row_for(("Signal", "inp_type"), ("Signal", "include_subdir"))
        row_for(("Signal", "img_dir"))                           # Directory
        row_for(("Signal", "File"))                              # Image File
        # File Type | Meta Type: Meta Type ~10% narrower with a tight label.
        row_for(("Signal", "img_ext"), ("Signal", "meta_ext"),
                stretches={("Signal", "img_ext"): 11, ("Signal", "meta_ext"): 9},
                tight={("Signal", "meta_ext")})
        row_for(("Signal", "Filter"))                            # Filter
        row_for(("Signal", "meta_dir"))                          # SPEC Dir
        # Anything else (e.g. NeXus File / Entry) one per row, in order.
        for field in fields:
            if field.path not in rendered:
                self._add_bound_row(self.source_card, field)

    def _render_experiment_bound_section(
        self,
        profile: ControlProfile,
        fields: tuple[ControlFormField, ...],
    ) -> None:
        if not fields:
            return
        # Producer actions (Calibrate / Refine / Make Mask) sit at the top of the
        # section they write into — co-located with §3 state, not a top bar.
        producers = self._experiment_producers(profile)
        if producers:
            row = QtWidgets.QWidget()
            row.setObjectName("controlsV2ActionRow")
            prow = QtWidgets.QHBoxLayout(row)
            prow.setContentsMargins(0, 0, 0, 3)
            prow.setSpacing(5)
            for spec in producers:
                btn = ActionButton(spec)
                btn.setText(_ACTION_LABELS.get(spec.action, spec.label))
                btn.actionRequested.connect(self.controlActionRequested)
                prow.addWidget(btn, 1)
            self.experiment_card.add_row(row)
        detector_paths = {
            ("Signal", "poni_file"),
            ("Calibration", "poni_file"),
            ("Signal", "mask_file"),
        }
        # Sample & measurement shows a Standard | Grazing segmented control (not a
        # group-header checkbox — that was the #56 repaint class).  The GI detail
        # fields (motor, value, orientation, tilt) render inline beneath it, but
        # only in Grazing mode: controls_logic gates their PRESENCE on the Grazing
        # state (progressive disclosure), so here they are simply rendered when
        # present.
        gi_paths = {
            ("GI", "Grazing"),
            ("GI", "th_motor"),
            ("GI", "th_val"),
            ("GI", "sample_orientation"),
            ("GI", "tilt_angle"),
        }
        groups = [
            (
                "Detector",
                tuple(field for field in fields if field.path in detector_paths),
                self._detector_status(fields, profile.detector_summary),
            ),
            (
                "Sample & measurement",
                tuple(field for field in fields if field.path in gi_paths),
                self._experiment_status(fields),
            ),
        ]
        # Subsection titles render in the section accent (amber) with NO number
        # prefix — consistent with every other section's subsections.
        for title, group_fields, status in groups:
            if not group_fields:
                continue
            group = SubsectionCard(title, status=status, accent="experiment")
            if title == "Sample & measurement":
                self._render_sample_measurement_rows(group, group_fields)
            else:
                for field in group_fields:
                    self._add_bound_row(group, field)
            self.experiment_card.add_row(group)

    def _render_sample_measurement_rows(
        self,
        group: "SubsectionCard",
        fields: tuple[ControlFormField, ...],
    ) -> None:
        """Render the Standard|Grazing segmented control, then a compact GI row:
        the θ motor combo, the manual θ value (manual mode only), and a '…'
        button that opens a small popup with the less-used Orientation + Tilt
        Angle options.  controls_logic drops the detail fields in Standard mode."""
        compact_labels = {
            ("GI", "th_motor"): "θ motor",
            ("GI", "th_val"): "θ",
        }
        popup_paths = {("GI", "sample_orientation"), ("GI", "tilt_angle")}
        inline = []
        popup_fields = []
        for field in fields:
            if field.path == ("GI", "Grazing"):
                seg = SegmentedControl(
                    field.path,
                    (("Standard", False), ("Grazing", True)),
                    value=bool(field.value),
                    enabled=bool(field.enabled),
                    reason=field.reason or "",
                )
                seg.valueChanged.connect(self.fieldValueChanged)
                group.add_row(seg)
            elif field.path in popup_paths:
                popup_fields.append(field)
            else:
                inline.append(field)  # θ motor, θ value (manual)
        if not (inline or popup_fields):
            return
        # θ motor + (manual) θ value share one row; Orientation + Tilt go behind
        # a '…' popup so the row isn't squished (Vivek).
        row = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        # θ motor (dropdown) takes ~2/3, the θ value box ~1/3 (≈30% narrower) so
        # the motor name isn't clipped; the motor combo expands to fill (Vivek).
        stretch = {("GI", "th_motor"): 2, ("GI", "th_val"): 1}
        for field in inline:
            sub = self._make_bound_row(
                field, label=compact_labels.get(field.path, field.label))
            if getattr(sub, "label", None) is not None:
                sub.label.setMinimumWidth(0)
            lay.addWidget(sub, stretch.get(field.path, 1))
        if popup_fields:
            more = QtWidgets.QToolButton()
            more.setText("…")
            more.setObjectName("controlsV2MoreButton")
            more.setToolTip("More GI options: Orientation, Tilt Angle")
            more.clicked.connect(
                lambda _=False, f=tuple(popup_fields): self._open_gi_more_popup(f))
            lay.addWidget(more, 0)
        group.add_row(row)

    def _close_gi_more_popup(self) -> None:
        """Close + dispose the GI '…' options popup if open, clearing the ref so a
        stale popup can't be harvested by ``current_form_edits`` (F1/F2)."""
        popup = getattr(self, "_gi_options_popup", None)
        if popup is not None:
            popup.close()
            # deleteLater() is async — its FormRow children would still be found
            # by findChildren() (current_form_edits) until the event loop runs.
            # setParent(None) detaches the subtree from this panel NOW, so a
            # rebuild can't harvest its stale rows before the deferred delete.
            popup.setParent(None)
            popup.deleteLater()
        self._gi_options_popup = None

    def _open_gi_more_popup(self, fields: tuple[ControlFormField, ...]) -> None:
        """Small floating popup for the less-used GI options (Orientation, Tilt
        Angle).  Its rows write through like any other V2 row."""
        self._close_gi_more_popup()
        popup = QtWidgets.QWidget(self, QtCore.Qt.Tool)
        popup.setObjectName("controlsV2GIMorePopup")
        popup.setWindowTitle("GI Options")
        lay = QtWidgets.QVBoxLayout(popup)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        for field in fields:
            lay.addWidget(self._make_bound_row(field))
        self._gi_options_popup = popup  # keep a ref so it isn't GC'd
        popup.show()
        popup.raise_()
        popup.activateWindow()

    def _render_processing_bound_section(
        self,
        profile: ControlProfile,
        fields: tuple[ControlFormField, ...],
    ) -> None:
        inline_fields = [field for field in fields if not field.parameter_group]
        advanced_fields = [field for field in fields if field.parameter_group]
        groups: dict[str, list[ControlFormField]] = {
            "1-D": [],
            "2-D": [],
            "Conditioning": [],
            "Background": [],
        }
        for field in inline_fields:
            groups.setdefault(self._processing_group_for_path(field.path), []).append(field)

        for name in ("1-D", "2-D", "Conditioning", "Background"):
            group_fields = groups.get(name) or []
            if not group_fields:
                continue
            # Every subsection title (1-D / 2-D / Conditioning / Background)
            # renders in the section accent (red) with no chip/number prefix —
            # consistent across all sections.
            sub = SubsectionCard(name, accent="processing",
                                 status=self._group_status(group_fields))
            self._render_processing_group_rows(sub, group_fields)
            self.processing_card.add_row(sub)

        if advanced_fields:
            sub = SubsectionCard(
                "Advanced",
                accent="processing",
                status=self._group_status(advanced_fields),
            )
            self._render_processing_group_rows(sub, advanced_fields)
            sub.set_collapsed(True)
            self.processing_card.add_row(sub)

        self._add_actions(
            self.processing_card,
            profile.actions_for(SectionId.PROCESSING),
            expand=True,
        )

    def _render_processing_group_rows(
        self,
        sub: SubsectionCard,
        fields: Sequence[ControlFormField],
    ) -> None:
        """Render a Processing group, coalescing low/high(/auto) triples and the
        Threshold min/max into one compact :class:`RangeRow` each (mockup)."""
        by_path = {field.path: field for field in fields}
        consumed: set[tuple[str, ...]] = set()
        # The point-count field(s) park on the Axis row, right of the dropdown —
        # NOT their own body row, and no longer in the subsection header (which
        # stays clear for a status line).  Collected here; attached when the Axis
        # row is built below.
        point_fields = [
            f for f in fields
            if f.path and f.path[-1] in (
                "points", "points_oop", "radial_points", "azim_points")
        ]
        consumed.update(f.path for f in point_fields)
        points_attached = False
        # Consecutive standalone bool toggles render as one compact pill row
        # (mockup), not full-width stacked buttons.
        pending_pills: list[ControlFormField] = []

        def flush_pills() -> None:
            if pending_pills:
                row = PillRow(list(pending_pills))
                row.valueChanged.connect(self.fieldValueChanged)
                sub.add_row(row)
                pending_pills.clear()

        for field in fields:
            path = field.path
            if path in consumed:
                continue
            last = path[-1] if path else ""
            kind = (
                field.kind.value
                if isinstance(field.kind, ControlFieldKind)
                else str(field.kind)
            )
            stem_paths = self._range_partner_paths(path)
            if stem_paths is not None:
                low_p, high_p, auto_p = stem_paths
                low_f, high_f = by_path.get(low_p), by_path.get(high_p)
                if low_f is not None and high_f is not None:
                    flush_pills()
                    sub.add_row(self._make_range_row(low_f, high_f, by_path.get(auto_p)))
                    consumed.update({low_p, high_p, auto_p})
                    continue
            # A bare _high / _auto whose _low partner is in this group is folded
            # into the RangeRow above; skip it.
            if last.endswith(("_high", "_auto")) and self._range_low_path(path) in by_path:
                consumed.add(path)
                continue
            # Threshold: (Mask, Threshold)=enable + (Mask, min) + (Mask, max).
            if path == ("Mask", "min") and ("Mask", "max") in by_path:
                flush_pills()
                sub.add_row(self._make_range_row(
                    field, by_path[("Mask", "max")], by_path.get(("Mask", "Threshold")),
                    label="Threshold"))
                consumed.update({("Mask", "min"), ("Mask", "max"), ("Mask", "Threshold")})
                continue
            if path in {("Mask", "max"), ("Mask", "Threshold")} and ("Mask", "min") in by_path:
                consumed.add(path)
                continue
            if kind == "bool":
                pending_pills.append(field)
                continue
            flush_pills()
            if last == "axis":
                # Drop the redundant "1D"/"2D" prefix and park the Pts cluster to
                # the right of the Axis dropdown (frees the header for status).
                row = self._make_bound_row(field, label="Axis")
                if point_fields:
                    self._attach_points_to_row(row, point_fields)
                    points_attached = True
                sub.add_row(row)
                continue
            self._add_bound_row(sub, field)
        flush_pills()
        # Safety net: never drop the point fields if a group has them but no
        # Axis row to host them.
        if point_fields and not points_attached:
            self._add_points_header(sub, point_fields)

    def _add_points_header(
        self,
        sub: SubsectionCard,
        point_fields: Sequence[ControlFormField],
    ) -> None:
        """Put the point-count field(s) on the subsection header as ``Pts [n] …``.

        Reuses :class:`FormRow` (label hidden) so the editors stay harvestable by
        ``current_form_edits`` and route through the same write-through path."""
        label = QtWidgets.QLabel("Pts")
        label.setObjectName("controlsV2HeaderLabel")
        sub.add_header_widget(label)
        for field in point_fields:
            row = FormRow(
                label="",
                path=field.path,
                value=field.value,
                kind="line",
                enabled=field.enabled,
                reason=field.reason,
            )
            row.label.hide()
            row.setMaximumWidth(72)
            row.valueChanged.connect(self.fieldValueChanged)
            sub.add_header_widget(row)

    @staticmethod
    def _range_low_path(path: tuple[str, ...]) -> tuple[str, ...]:
        stem = path[-1].rsplit("_", 1)[0]
        return path[:-1] + (f"{stem}_low",)

    @staticmethod
    def _range_partner_paths(path: tuple[str, ...]):
        """For a ``*_low`` path return (low, high, auto) sibling paths, else None."""
        if not path or not path[-1].endswith("_low"):
            return None
        stem = path[-1][:-len("_low")]
        base = path[:-1]
        return path, base + (f"{stem}_high",), base + (f"{stem}_auto",)

    def _make_range_row(
        self,
        low_field: ControlFormField,
        high_field: ControlFormField,
        toggle_field: ControlFormField | None,
        *,
        label: str | None = None,
    ) -> "RangeRow":
        if label is None:
            label = low_field.label
            for suffix in (" Low", " Range"):
                if label.endswith(suffix):
                    label = label[: -len(suffix)]
                    break
        toggle = None
        if toggle_field is not None:
            toggle = {
                "path": toggle_field.path,
                "value": toggle_field.value,
                "enabled": toggle_field.enabled,
                "tooltip": toggle_field.reason or "Auto",
            }
        row = RangeRow(
            label=label,
            low={"path": low_field.path, "value": low_field.value,
                 "enabled": low_field.enabled},
            high={"path": high_field.path, "value": high_field.value,
                  "enabled": high_field.enabled},
            toggle=toggle,
        )
        row.valueChanged.connect(self.fieldValueChanged)
        return row

    @staticmethod
    def _experiment_status(fields: tuple[ControlFormField, ...]) -> str:
        values = {field.path: field.value for field in fields}
        if values.get(("GI", "Grazing")):
            return "grazing"
        return "standard"

    @staticmethod
    def _detector_status(
        fields: tuple[ControlFormField, ...],
        detector_summary: str = "",
    ) -> str:
        if detector_summary:
            return detector_summary
        values = {field.path: field.value for field in fields}
        if values.get(("Signal", "poni_file")) or values.get(("Calibration", "poni_file")):
            return "fitted"
        if values.get(("Signal", "mask_file")):
            return "mask"
        return ""

    @staticmethod
    def _source_status(profile: ControlProfile) -> str:
        fields = profile.fields
        frame_status = fields.get(FieldId.SOURCE_FRAMES)
        raw_status = fields.get(FieldId.SOURCE_RAW)
        parts = []
        if frame_status is not None and frame_status.value:
            parts.append(f"{frame_status.value} frames")
        if raw_status is not None and raw_status.value:
            parts.append(raw_status.value)
        return " · ".join(parts)

    @staticmethod
    def _processing_status(profile: ControlProfile) -> str:
        try:
            return str(profile.processing_page.value).replace("_", " ")
        except Exception:
            return ""

    @staticmethod
    def _group_status(fields: Sequence[ControlFormField]) -> str:
        disabled = sum(1 for field in fields if not field.enabled)
        if disabled and disabled == len(fields):
            return "locked"
        return ""

    def _clear_layout(self, layout: QtWidgets.QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

    _PRODUCER_ORDER = {
        ControlAction.CALIBRATE: 0,
        ControlAction.REFINE_GEOMETRY: 1,
        ControlAction.MAKE_MASK: 2,
    }

    def _render_top_actions(self, profile: ControlProfile) -> None:
        # Producer actions (Calibrate / Refine / Make Mask) now render INSIDE the
        # Experiment section (they write §3 state), not a separate top bar.
        self._clear_layout(self.top_action_layout)
        self.top_action_bar.hide()

    def _experiment_producers(self, profile: ControlProfile):
        """The instrument-producer actions, sorted Calibrate · Refine · Make Mask."""
        return sorted(
            profile.actions_for(SectionId.EXPERIMENT),
            key=lambda spec: self._PRODUCER_ORDER.get(spec.action, 99),
        )

    def _make_bound_row(
        self,
        field: ControlFormField,
        *,
        label: str | None = None,
    ) -> FormRow:
        row = FormRow(
            label=field.label if label is None else label,
            path=field.path,
            value=field.value,
            kind=(
                field.kind.value
                if isinstance(field.kind, ControlFieldKind)
                else str(field.kind)
            ),
            choices=field.choices,
            browse=field.browse,
            enabled=field.enabled,
            reason=field.reason,
        )
        row.valueChanged.connect(self.fieldValueChanged)
        row.browseRequested.connect(self.fieldBrowseRequested)
        if field.browse and field.value and not field.reason:
            # Path/file fields truncate in the narrow panel; show the FULL path
            # on hover (a disabled field still shows its reason).
            tip = str(field.value)
        else:
            tip = _field_tooltip(field.path, field.reason)
        if tip:
            # Set on the editor + label too (not just the container) so the
            # tooltip shows on hover over the control itself.  The browse button
            # keeps its own "Browse …" tooltip.
            for widget in (row, row.editor, row.label):
                if widget is not None:
                    widget.setToolTip(tip)
        return row

    def _add_bound_row(
        self,
        card: SectionCard | SubsectionCard,
        field: ControlFormField,
        *,
        label: str | None = None,
    ) -> None:
        card.add_row(self._make_bound_row(field, label=label))

    def _attach_points_to_row(
        self,
        row: FormRow,
        point_fields: Sequence[ControlFormField],
    ) -> None:
        """Park the compact ``Pts [n] …`` cluster on the right of an Axis row.

        Reuses :class:`FormRow` (label hidden) so the editors stay harvestable by
        ``current_form_edits`` and route through the same write-through path."""
        label = QtWidgets.QLabel("Pts")
        label.setObjectName("controlsV2HeaderLabel")
        row.add_trailing_widget(label)
        for field in point_fields:
            pr = FormRow(
                label="",
                path=field.path,
                value=field.value,
                kind="line",
                enabled=field.enabled,
                reason=field.reason,
            )
            pr.label.hide()
            pr.setMaximumWidth(58)   # ~20% narrower than the old 72px cap
            pr.valueChanged.connect(self.fieldValueChanged)
            row.add_trailing_widget(pr)

    @staticmethod
    def _processing_group_for_path(path: tuple[str, ...]) -> str:
        if not path:
            return ""
        # Average Scan is a wrangler bool re-homed into PROCESSING; match the
        # full path because the "Signal" root carries many non-processing params.
        if path == ("Signal", "series_average"):
            return "Conditioning"
        root = path[0]
        if root in {"Mask", "MaskSat"}:
            return "Conditioning"
        if root == "Int1D":
            return "1-D"
        if root == "Int2D":
            return "2-D"
        if root == "BG":
            return "Background"
        return ""

    def _add_actions(
        self,
        card: SectionCard,
        actions: tuple[ControlActionSpec, ...],
        *,
        expand: bool = False,
    ) -> None:
        if not actions:
            return
        row = QtWidgets.QWidget()
        row.setObjectName("controlsV2ActionRow")
        lay = QtWidgets.QHBoxLayout(row)
        lay.setContentsMargins(0, 2, 0, 0)
        lay.setSpacing(5)
        if not expand:
            lay.addStretch(1)
        for spec in actions:
            btn = ActionButton(spec)
            btn.actionRequested.connect(self.controlActionRequested)
            lay.addWidget(btn, 1 if expand else 0)
        card.add_row(row)

    def _render_analysis(self, launchers: tuple[AnalysisLauncherSpec, ...]) -> None:
        self.analysis_card.clear_rows()
        if self._bound_state is not None:
            self.analysis_card.hide()
            return
        self.analysis_card.show()
        for status in (self._profile.fields_for(SectionId.ANALYSIS)
                       if self._profile is not None else ()):
            self.analysis_card.add_row(FieldRow(status))
        for spec in launchers:
            btn = LauncherButton(spec)
            btn.launched.connect(self.analysisLaunchRequested)
            self.analysis_card.add_row(btn)
