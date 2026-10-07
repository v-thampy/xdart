# -*- coding: utf-8 -*-
"""Shared Controls renderer for immutable readiness snapshots."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import PurePath

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.themes.spacing import SpacingTokens, current_spacing_tokens

from xrd_tools.session.readiness import (
    ControlAction,
    ControlActionSpec,
    ControlFieldKind,
    ControlFormEdit,
    ControlFormField,
    ControlsProjection,
    ProcessingPage,
    SectionId,
    integer_field_refusal,
)


def _spaced(base: int, tokens: SpacingTokens) -> int:
    """Scale a base layout value while keeping Normal byte-compatible."""
    return max(0, base + tokens.layout_gap - 8)


PROCESSING_ROW_LABEL_WIDTH = 72
PROCESSING_ROW_GAP = 5
PROCESSING_PILL_TOP_MARGIN = 3


# Order: Project · Experiment · Source · Processing (instrument config before
# data, matching the notebook workflow, Vivek).  The number follows the order;
# the accent follows the section identity.
_SECTION_META = {
    SectionId.PROJECT: ("1", "PROJECT", "project"),
    SectionId.EXPERIMENT: ("2", "EXPERIMENT", "experiment"),
    SectionId.SOURCE: ("3", "SOURCE", "source"),
    SectionId.PROCESSING: ("4", "PROCESSING", "processing"),
}

_ACTION_LABELS = {
    ControlAction.CALIBRATE: "⌖ Calibrate",
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
    ("Signal", "series_average"): (
        "Average all frames in the series into one frame before integration."),
    ("Mask", "Threshold"): (
        "Exclude values outside the inclusive [min, max] band per frame. "
        "Automatic: values at or above the detector's saturation are left out. "
        "Off keeps all finite values. The global mask always applies."),
    ("Mask", "min"): (
        "Manual threshold lower bound (pixels below are masked). Default 0."),
    ("Mask", "max"): (
        "Upper threshold (pixels above are masked). Automatic upper limit: the "
        "detector's saturation (the value the input records, else the "
        "calibration detector's counter limit, else the data type's maximum), "
        "never the frame maximum. Clear this bound to restore the automatic band."),
    # Source
    ("Signal", "inp_type"): (
        "Source kind: a numbered image series, a directory of images, or a "
        "single image."),
    ("Signal", "include_subdir"): "Also search sub-directories for matching images.",
    ("Signal", "Filter"): (
        "Filename filter. Space-separated terms must ALL match "
        "(case-insensitive substrings, any order): 'lab6 cal'. "
        "Alternatives with | or OR: 'scan1 | scan2'. "
        "Exclude with -term or NOT term: 'scan -bg'. "
        "Empty matches every file."),
    ("Signal", "img_ext"): "Image file type (extension).",
    ("Signal", "meta_ext"): "Per-frame metadata format (auto, none, txt, metadata, spec, pdi).",
    ("Signal", "meta_dir"): (
        "Directory holding the SPEC metadata file, when it's separate from the "
        "images."),
}

# Descriptive hover tooltips for the producer/inspector action buttons.
_ACTION_TOOLTIPS: dict[ControlAction, str] = {
    ControlAction.CALIBRATE: "Run pyFAI calibration to produce a PONI (detector geometry) file.",
    ControlAction.MAKE_MASK: "Build a detector mask from a chosen TIFF image.",
    ControlAction.ADVANCED_PROCESSING: "Open the advanced integration settings (full parameter tree).",
    ControlAction.REINTEGRATE_1D: "Re-integrate the loaded scan to 1D with the current settings.",
    ControlAction.REINTEGRATE_2D: "Re-integrate the loaded scan to 2D with the current settings.",
}


def _field_tooltip(path, reason: str = "") -> str:
    """Resolve a control's hover tooltip: the disabled `reason` takes precedence
    (so a locked control explains WHY), else the descriptive help text."""
    return (reason or "") or _FIELD_TOOLTIPS.get(tuple(path), "")


_SOURCE_ENERGY_PATH = ("Source", "energy_preference")
_SOURCE_ENERGY_OPTIONS = (("PONI", "poni"), ("Metadata", "metadata"))
# Maintainer policy (2026-08-04, exception-based — supersedes the old
# enumerated allowlist): ONLY the project folder displays its full path while
# idle (it anchors the whole session).  Every other browse-backed path field
# displays its basename / final directory component while idle, expands to its
# exact full model value while focused for direct editing, and retains that
# full value in the tooltip and committed model throughout.
_FULL_PATH_DISPLAY_PATHS = {
    ("Project", "project_folder"),
}
_PROJECT_ROOT_PATH = ("Project", "project_folder")


class ActionButton(QtWidgets.QPushButton):
    """Small card action button carrying a ``ControlActionSpec``."""

    actionRequested = QtCore.Signal(object)

    def __init__(self, spec: ControlActionSpec, parent=None):
        super().__init__(spec.label, parent)
        self.setObjectName("controlsActionButton")
        self._spec = spec
        self.clicked.connect(lambda: self.actionRequested.emit(self._spec.action))
        self.apply_spec(spec)

    @property
    def spec(self) -> ControlActionSpec:
        return self._spec

    def apply_spec(self, spec: ControlActionSpec) -> None:
        label = (
            spec.label
            if spec.label.startswith("Cancel ")
            else _ACTION_LABELS.get(spec.action, spec.label)
        )
        enabled = bool(spec.enabled)
        tooltip = spec.reason or _ACTION_TOOLTIPS.get(spec.action, "")
        production_ready = bool(spec.production_ready)
        # Role drives a subtle tint: Reintegrate = green (run-like), Advanced =
        # red (the destructive/expert escape hatch); producers stay neutral.
        if spec.action in (ControlAction.REINTEGRATE_1D, ControlAction.REINTEGRATE_2D):
            role = "reintegrate"
        elif spec.action == ControlAction.ADVANCED_PROCESSING:
            role = "advanced"
        else:
            role = ""
        style_changed = (
            self.property("productionReady") != production_ready
            or self.property("actionRole") != role
        )
        if (
            self._spec == spec
            and self.text() == label
            and self.isEnabled() == enabled
            and self.toolTip() == tooltip
            and not style_changed
        ):
            return

        self._spec = spec
        if self.text() != label:
            self.setText(label)
        if self.isEnabled() != enabled:
            self.setEnabled(enabled)
        if self.toolTip() != tooltip:
            self.setToolTip(tooltip)
        if self.property("productionReady") != production_ready:
            self.setProperty("productionReady", production_ready)
        if self.property("actionRole") != role:
            self.setProperty("actionRole", role)
        if style_changed:
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
        compact: bool = False,
    ):
        super().__init__(parent)
        self.setObjectName("controlsSectionCard")
        self.setProperty("accent", accent)
        self._collapsed = False
        self._collapsible = bool(collapsible)
        self._compact = bool(compact)
        outer = QtWidgets.QVBoxLayout(self)
        self._outer_layout = outer
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.header = QtWidgets.QFrame()
        self.header.setObjectName("controlsSectionHeader")
        self.header.setProperty("accent", accent)
        header_lay = QtWidgets.QHBoxLayout(self.header)
        self._header_layout = header_lay
        header_lay.setContentsMargins(7, 5, 8, 5)
        header_lay.setSpacing(7)
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setObjectName("controlsChevron")
        self.toggle.setText("▾")
        self.toggle.setAutoRaise(True)
        self.toggle.setToolTip("Collapse section")
        self.toggle.setEnabled(self._collapsible)
        self.toggle.clicked.connect(self.toggle_collapsed)
        header_lay.addWidget(self.toggle)
        self.chip = QtWidgets.QLabel(number)
        self.chip.setObjectName("controlsSectionChip")
        self.chip.setProperty("accent", accent)
        self.chip.setAlignment(QtCore.Qt.AlignCenter)
        self.chip.setVisible(bool(number))
        header_lay.addWidget(self.chip)
        self.title = QtWidgets.QLabel(title.upper())
        self.title.setObjectName("controlsSectionTitle")
        header_lay.addWidget(self.title, 1)
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("controlsSectionStatus")
        # Status detail uses the available header space, never the column's width floor.
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                  QtWidgets.QSizePolicy.Policy.Preferred)
        self.status.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.status.hide()
        header_lay.addWidget(self.status, 1)
        self.valid_marker = QtWidgets.QLabel("✓")
        self.valid_marker.setObjectName("controlsSectionTick")
        self.valid_marker.setProperty("accent", accent)
        self.valid_marker.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.valid_marker.hide()
        header_lay.addWidget(self.valid_marker)
        self.header.mousePressEvent = self._header_mouse_press
        outer.addWidget(self.header)

        self.body = QtWidgets.QFrame()
        self.body.setObjectName("controlsSectionBody")
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(8, 7, 8, 8)
        self.body_layout.setSpacing(5)
        outer.addWidget(self.body)

        self.embedded = QtWidgets.QFrame()
        self.embedded.setObjectName("controlsEmbedded")
        self.embedded_layout = QtWidgets.QVBoxLayout(self.embedded)
        self.embedded_layout.setContentsMargins(0, 4, 0, 0)
        self.embedded_layout.setSpacing(0)
        self.embedded.hide()
        self.body_layout.addWidget(self.embedded)
        if self._compact:
            self.setObjectName("controlsInlineReadiness")
            self.header.setObjectName("controlsInlineReadinessHeader")
            for widget in (self.toggle, self.chip, self.title, self.body):
                widget.hide()
            header_lay.setStretchFactor(self.status, 1)
            self.status.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self._apply_spacing(current_spacing_tokens())

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in {
            QtCore.QEvent.Type.StyleChange,
            QtCore.QEvent.Type.FontChange,
        } and hasattr(self, "embedded_layout"):
            self._apply_spacing(current_spacing_tokens())

    def _apply_spacing(self, tokens: SpacingTokens) -> None:
        self._header_layout.setContentsMargins(
            0 if self._compact else _spaced(7, tokens),
            0 if self._compact else _spaced(5, tokens),
            0 if self._compact else _spaced(8, tokens),
            0 if self._compact else _spaced(5, tokens),
        )
        self._header_layout.setSpacing(_spaced(7, tokens))
        self.body_layout.setContentsMargins(
            _spaced(8, tokens),
            _spaced(7, tokens),
            _spaced(8, tokens),
            _spaced(8, tokens),
        )
        self.body_layout.setSpacing(_spaced(5, tokens))
        self.embedded_layout.setContentsMargins(
            0, _spaced(4, tokens), 0, 0
        )

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
        if self.status.text() == text and self.status.isVisible() == bool(text):
            return
        self.status.setText(text)
        self.status.setVisible(bool(text))

    def set_valid_marker(self, visible: bool, tooltip: str = "") -> None:
        visible = bool(visible)
        tooltip = tooltip or ""
        if (
            self.valid_marker.isVisible() == visible
            and self.valid_marker.toolTip() == tooltip
        ):
            return
        self.valid_marker.setVisible(visible)
        self.valid_marker.setToolTip(tooltip)

    def set_readiness(self, ready: bool, text: str) -> None:
        """A passive readiness header, using the shared green/red section accents."""
        self.status.setWordWrap(True)
        self.set_status_text(text)
        self.valid_marker.setText("✓" if ready else "!")
        self.valid_marker.setProperty("accent", "source" if ready else "processing")
        self.set_valid_marker(True, text)
        self.valid_marker.style().unpolish(self.valid_marker)
        self.valid_marker.style().polish(self.valid_marker)

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
        label.setObjectName("controlsGroupHeader")
        self.add_row(label)

    def embedded_visible(self) -> bool:
        """The embedded widget's EXPLICIT projection state (§21.5 req 2).

        The visibility this card last actually applied — observed projection
        state, not the caller's desired configuration, and independent of whether
        an ancestor happens to be shown (so it is meaningful offscreen).  The
        source-recovery receipt proves against this."""
        return bool(getattr(self, "_embedded_visible", False))

    def set_embedded_widget(
        self,
        widget: QtWidgets.QWidget | None,
        *,
        visible: bool = True,
    ) -> None:
        self._embedded_visible = bool(widget is not None and visible)
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
        self.setObjectName("controlsSubsectionCard")
        self._collapsed = False
        outer = QtWidgets.QVBoxLayout(self)
        self._outer_layout = outer
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.header = QtWidgets.QFrame()
        self.header.setObjectName("controlsSubsectionHeader")
        h_lay = QtWidgets.QHBoxLayout(self.header)
        self._header_layout = h_lay
        h_lay.setContentsMargins(7, 4, 7, 4)
        h_lay.setSpacing(6)
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setObjectName("controlsSubChevron")
        self.toggle.setText("▾")
        self.toggle.setAutoRaise(True)
        self.toggle.clicked.connect(self.toggle_collapsed)
        h_lay.addWidget(self.toggle)
        if prefix:
            self.prefix = QtWidgets.QLabel(prefix)
            self.prefix.setObjectName("controlsSubsectionPrefix")
            # The prefix chip takes the parent section's accent (amber for
            # Experiment, violet for Processing, …), not the global purple.
            self.prefix.setProperty("accent", accent)
            h_lay.addWidget(self.prefix)
        else:
            self.prefix = None
        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("controlsSubsectionTitle")
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
        self.header_extra.setObjectName("controlsSubsectionHeaderExtra")
        self.header_extra_layout = QtWidgets.QHBoxLayout(self.header_extra)
        self.header_extra_layout.setContentsMargins(0, 0, 0, 0)
        self.header_extra_layout.setSpacing(5)
        self.header_extra.hide()
        h_lay.addWidget(self.header_extra)
        self.status = QtWidgets.QLabel(status)
        self.status.setObjectName("controlsSubsectionStatus")
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                  QtWidgets.QSizePolicy.Policy.Preferred)
        self.status.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.status.setVisible(bool(status))
        h_lay.addWidget(self.status, 1)
        self.header.mousePressEvent = self._header_mouse_press
        outer.addWidget(self.header)
        self.body = QtWidgets.QFrame()
        self.body.setObjectName("controlsSubsectionBody")
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(7, 5, 7, 7)
        self.body_layout.setSpacing(4)
        outer.addWidget(self.body)
        self._apply_spacing(current_spacing_tokens())

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in {
            QtCore.QEvent.Type.StyleChange,
            QtCore.QEvent.Type.FontChange,
        } and hasattr(self, "body_layout"):
            self._apply_spacing(current_spacing_tokens())

    def _apply_spacing(self, tokens: SpacingTokens) -> None:
        self._header_layout.setContentsMargins(
            _spaced(7, tokens),
            _spaced(4, tokens),
            _spaced(7, tokens),
            _spaced(4, tokens),
        )
        self._header_layout.setSpacing(_spaced(6, tokens))
        self.header_extra_layout.setSpacing(_spaced(5, tokens))
        self.body_layout.setContentsMargins(
            _spaced(7, tokens),
            _spaced(5, tokens),
            _spaced(7, tokens),
            _spaced(7, tokens),
        )
        self.body_layout.setSpacing(_spaced(4, tokens))

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


class _ComboPopupItemDelegate(QtWidgets.QStyledItemDelegate):
    """Keep native combo popup rows readable at compact spacing tiers.

    The platform popup view does not consistently honor an item ``min-height``
    from QSS.  Deriving the floor from the live closed control keeps the popup
    responsive to font/spacing changes without baking another appearance
    constant into Controls.
    """

    def __init__(self, combo: QtWidgets.QComboBox):
        super().__init__(combo)
        self._combo = combo

    def sizeHint(self, option, index) -> QtCore.QSize:
        size = super().sizeHint(option, index)
        control_height = max(
            self._combo.height(),
            self._combo.sizeHint().height(),
        )
        size.setHeight(max(size.height(), control_height))
        return size


class _ControlsComboBox(QtWidgets.QComboBox):
    """Combo whose popup capacity survives Controls form reconstruction.

    On macOS the styled popup container can retain the closed control's Tight
    height after a mounted form is rebuilt.  The item delegate still gives
    every row the right height, but the viewport then shows only one or two
    rows plus native scroll affordances.  Recompute the viewport floor at the
    only authoritative boundary -- immediately before the popup is shown.
    """

    def showPopup(self) -> None:
        view = self.view()
        view.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        view.doItemsLayout()
        popup_width = (
            view.sizeHintForColumn(0) + 2 * view.frameWidth()
            + view.style().pixelMetric(
                QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent, None, view,
            )
        )
        view.setMinimumWidth(max(self.width(), popup_width))
        visible = min(self.count(), max(1, self.maxVisibleItems()))
        row_height = max(
            (view.sizeHintForRow(row) for row in range(visible)),
            default=max(self.height(), self.sizeHint().height()),
        )
        # Include the view frame without imposing a maximum: longer choice
        # inventories retain native scrolling after ``maxVisibleItems``.
        viewport_floor = visible * row_height + 2 * view.frameWidth()
        view.setMinimumHeight(max(1, viewport_floor))
        view.updateGeometry()
        super().showPopup()


class ControlsComboBox(_ControlsComboBox):
    """The Controls combo for any host: popup rows at the closed control's height, never elided.

    Qt's native popup ignores the QSS item-height floor on some platforms; the
    parented delegate keeps a Tight live-theme change from compressing choices.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("controlsComboBox")
        self.setItemDelegate(_ComboPopupItemDelegate(self))


def browse_button(label: str) -> QtWidgets.QToolButton:
    """The Controls 📁 browse button; the host connects ``clicked``."""
    button = QtWidgets.QToolButton()
    button.setText("📁")
    button.setObjectName("controlsBrowseButton")
    button.setToolTip(f"Browse {label}")
    button.setMinimumWidth(31)
    button.setMaximumWidth(36)
    return button


class IntegerLineEdit(QtWidgets.QLineEdit):
    """A whole-number row's editor (``FormRow`` with ``kind="int"``): a line edit whose text is kept as typed and
    shown refused, never reverted, with a spin box's ``value``/``setValue`` for a window that drives the row itself
    (FIT-TOOLS-2026-10: Int's Peak Fit sets its peak count to the peaks it found)."""

    def value(self) -> int | None:
        """The text as a whole number (``integer_field_refusal``'s reading), None when it is blank or none (the row
        says why: ``FormRow.refusal``)."""
        try:
            return int(self.text().strip())
        except ValueError:
            return None

    def setValue(self, value: int | None) -> None:  # noqa: N802 (a spin box's name, which its callers use)
        """Show ``value`` (None: blank), as code sets text: no draft and no commit."""
        self.setText("" if value is None else str(int(value)))


class FormRow(QtWidgets.QWidget):
    """One editable row in the Controls form."""

    valueChanged = QtCore.Signal(object, object)
    #: A still-uncommitted USER draft (line-editor ``textEdited`` — never a
    #: programmatic ``setText``), so a focused draft can be revisioned at action
    #: time (§12.4) without being applied until it commits.
    draftChanged = QtCore.Signal(object, object)
    browseRequested = QtCore.Signal(object)

    def __init__(
        self,
        *,
        label: str,
        path: tuple[str, ...],
        value,
        kind: str = "line",
        choices: Sequence[str] = (),
        choice_labels: Mapping[str, str] | None = None,
        browse: bool = False,
        enabled: bool = True,
        reason: str = "",
        minimum: int | None = None,
        maximum: int | None = None,
        allow_blank: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("controlsFormRow")
        self._path = tuple(path)
        self._kind = kind
        self._choice_labels = dict(choice_labels or {})
        self._enabled = bool(enabled)
        #: "int" rows: the bounds their text is checked against (``integer_field_refusal``)
        self._bounds = (minimum, maximum, bool(allow_blank))
        self._reason = reason
        self._purpose_tooltip = ""
        self._browse = bool(browse)
        self._file_name_only = (
            kind == "line"
            and browse
            and self._path not in _FULL_PATH_DISPLAY_PATHS
        )
        self._model_value = "" if value is None else str(value)
        self._editor_dirty = False
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.label = QtWidgets.QLabel(label)
        self.label.setObjectName("controlsFieldLabel")
        self.label.setMinimumWidth(76)

        if kind == "bool":
            # The Qt-backed integrator panel uses checkable QPushButtons for
            # GI/threshold/saturation toggles.  Mirroring that behavior keeps
            # checked-but-disabled controls visibly checked during active runs.
            self.label.hide()
            editor = QtWidgets.QPushButton(label)
            editor.setObjectName("controlsToggleButton")
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
            editor = ControlsComboBox()
            self._combo_popup_delegate = editor.itemDelegate()
            values = [str(choice) for choice in choices]
            if not values and value not in (None, ""):
                values = [str(value)]
            if self._choice_labels:
                for item in values:
                    editor.addItem(self._choice_labels.get(item, item), item)
            else:
                editor.addItems(values)
            text = "" if value is None else str(value)
            idx = editor.findData(text) if self._choice_labels else editor.findText(text)
            if idx >= 0:
                editor.setCurrentIndex(idx)
            elif text:
                if self._choice_labels:
                    editor.addItem(text, text)
                else:
                    editor.addItem(text)
                editor.setCurrentIndex(editor.count() - 1)
            # ``currentTextChanged`` also fires for programmatic reconciliation,
            # and rebuilding the passive Controls form synchronously from that
            # signal can destroy a native popup while it is still open.  Accept
            # only a genuine user choice and commit it on the next event-loop
            # turn, after the popup has closed.
            editor.textActivated.connect(
                lambda text: QtCore.QTimer.singleShot(
                    0,
                    lambda selected=str(editor.currentData() if self._choice_labels else text): self._emit_edit(selected),
                )
            )
            lay.addWidget(editor, 1)
        else:
            lay.addWidget(self.label)
            editor = (IntegerLineEdit if kind == "int" else QtWidgets.QLineEdit)(self._line_display_value(value))
            editor.setObjectName("controlsLineEdit")
            editor.editingFinished.connect(self._emit_editor_edit)
            # A focused draft (user typing, NOT programmatic setText) is
            # revisioned at action time so it survives a rebuild and wins by
            # revision (§12.4); it commits (and applies) on editingFinished.
            editor.textEdited.connect(self._draft_edited)
            if kind == "int":  # the typed text is kept and shown refused, never reverted
                editor.textChanged.connect(lambda _text: self._show_validity())
            lay.addWidget(editor, 1)

        self.editor = editor
        self.editor.setEnabled(self._enabled)
        if self._file_name_only:
            self.editor.installEventFilter(self)
            # connected first, so a host's own ``textChanged`` slot reads the adopted value
            self.editor.textChanged.connect(self._adopt_set_text)
        # Path/browse fields should show the root of long paths by default; the
        # full value is still available on hover via the row tooltip.
        if browse and isinstance(self.editor, QtWidgets.QLineEdit):
            self.editor.setCursorPosition(0)
        self.browse_button = None
        if browse:
            btn = browse_button(label)
            btn.clicked.connect(lambda _=False, p=self._path: self.browseRequested.emit(p))
            btn.setEnabled(self._enabled)
            lay.addWidget(btn)
            self.browse_button = btn
        self._refresh_tooltip()
        if kind == "int":
            self._show_validity()

    @property
    def path(self) -> tuple[str, ...]:
        return self._path

    def set_purpose_tooltip(self, text: str) -> None:
        """Keep descriptive help separate from temporary refusal/disabled reasons."""
        self._purpose_tooltip = str(text)
        if self._kind == "int":
            self._show_validity()
        else:
            self._refresh_tooltip()

    def _refresh_tooltip(self, refusal: str = "") -> None:
        tip = (f"{self.label.text()}: {refusal}" if refusal else
               self._reason or (self._model_value if self._browse and self._model_value else
                                self._purpose_tooltip or _field_tooltip(self._path)))
        for widget in (self, self.editor, self.label):
            if widget.toolTip() != tip:
                widget.setToolTip(tip)

    def refusal(self) -> str:
        """Why an ``int`` row's text is no whole number within its bounds ("" when it is, and for
        every other kind): the rule a page's readiness names (``integer_field_refusal``)."""
        if self._kind != "int":
            return ""
        minimum, maximum, allow_blank = self._bounds
        return integer_field_refusal(self.editor.text(), minimum=minimum, maximum=maximum,
                                     allow_blank=allow_blank)

    def set_integer_bounds(self, minimum: int | None, maximum: int | None) -> None:
        """Change a whole-number row's bounds without changing its entered text."""
        if self._kind != "int":
            raise TypeError("integer bounds require a whole-number row")
        self._bounds = (minimum, maximum, self._bounds[2])
        self._show_validity()

    def _show_validity(self) -> None:
        """An ``int`` row marks refused text (the theme's ``invalid`` property) and says why in its
        tooltip; the text itself stays as typed."""
        refusal = self.refusal()
        if bool(self.editor.property("invalid")) != bool(refusal):
            self.editor.setProperty("invalid", bool(refusal))
            style = self.editor.style()
            style.unpolish(self.editor)
            style.polish(self.editor)
        self._refresh_tooltip(refusal)

    def add_trailing_widget(self, widget: QtWidgets.QWidget, stretch: int = 0) -> None:
        """Append a widget to the right of the row (after the editor).  Used to
        park the compact 'Pts' cluster on the Axis row — the editor stretches, so
        trailing widgets pin to the right; a ``stretch`` shares the width instead
        (a second field on a file row, e.g. the calibration window's image stem)."""
        self.layout().addWidget(widget, stretch)

    def current_value(self):
        if self._kind == "bool":
            return bool(self.editor.isChecked())
        if self._kind == "combo":
            return self.editor.currentData() if self._choice_labels else self.editor.currentText()
        if self._file_name_only and not self._editor_dirty:
            return self._model_value
        return self.editor.text()

    def apply_field(self, field: ControlFormField) -> bool:
        if tuple(field.path) != self._path:
            return False
        kind = field.kind.value if isinstance(field.kind, ControlFieldKind) else str(field.kind)
        if kind != self._kind:
            return False

        self._enabled = bool(field.enabled)
        self.editor.setEnabled(self._enabled)
        if self.browse_button is not None:
            self.browse_button.setEnabled(self._enabled)

        value = "" if field.value is None else str(field.value)
        was_blocked = self.editor.blockSignals(True)
        try:
            if self._kind == "bool":
                checked = bool(field.value)
                if self.editor.isChecked() != checked:
                    self.editor.setChecked(checked)
            elif self._kind == "combo":
                # The projected ``field.choices`` are the authoritative combo
                # vocabulary; construction-time items must not outlive them.
                # An explicit current value absent from the vocabulary is
                # appended last for representability only — never invented,
                # never reordered into the typed choices.
                desired = [str(choice) for choice in field.choices]
                if value and value not in desired:
                    desired.append(value)
                current_items = [
                    self.editor.itemData(index) if self._choice_labels else self.editor.itemText(index)
                    for index in range(self.editor.count())
                ]
                if current_items != desired:
                    self.editor.clear()
                    if self._choice_labels:
                        for item in desired:
                            self.editor.addItem(self._choice_labels.get(item, item), item)
                    else:
                        self.editor.addItems(desired)
                if self.current_value() != value:
                    idx = self.editor.findData(value) if self._choice_labels else self.editor.findText(value)
                    if idx >= 0:
                        self.editor.setCurrentIndex(idx)
            else:
                focused = self.editor.hasFocus()
                if not focused:
                    shown = self._line_display_value(value)
                    if self.editor.text() != shown:
                        self.editor.setText(shown)
                    if field.browse:
                        self.editor.setCursorPosition(0)
                # Once editingFinished has emitted, an immediate synchronous
                # owner projection is authoritative even though QLineEdit is
                # still completing its FocusOut event.  A rejected edit can
                # therefore restore the prior exact path before the deferred
                # basename collapse; an actively typed draft remains intact.
                if not focused or not self._editor_dirty:
                    self._model_value = value
                    self._editor_dirty = False
        finally:
            self.editor.blockSignals(was_blocked)

        self._reason = field.reason
        self._browse = bool(field.browse)
        if self._kind == "int":
            self._bounds = (field.minimum, field.maximum, bool(field.allow_blank))
            self._show_validity()
        else:
            self._refresh_tooltip()
        return True

    def _line_display_value(self, value) -> str:
        text = "" if value is None else str(value)
        if self._file_name_only and text:
            return PurePath(text).name
        return text

    def eventFilter(self, watched, event) -> bool:
        if watched is self.editor and self._file_name_only:
            event_type = event.type()
            if event_type == QtCore.QEvent.Type.FocusIn:
                # Presentation-only expansion: programmatic setText must not
                # manufacture a draft or commit.  The exact model value, not
                # the idle basename, is the starting point for direct edits.
                if not self._editor_dirty:
                    blocked = self.editor.blockSignals(True)
                    try:
                        self.editor.setText(self._model_value)
                    finally:
                        self.editor.blockSignals(blocked)
            elif event_type == QtCore.QEvent.Type.FocusOut:
                # QLineEdit emits editingFinished from its own focus-out
                # handler.  Collapse on the next turn so the full typed value
                # is committed first and never replaced by its basename.
                QtCore.QTimer.singleShot(0, self._collapse_path_editor)
        return super().eventFilter(watched, event)

    def _adopt_set_text(self, text: str) -> None:
        """A file row's text set by code while it is unfocused is the row's value: a window that drives
        its rows directly rather than through a projection (the calibration window) reads back what it
        set. It is shown collapsed, with the full path as the tooltip, and makes no draft and no commit
        (§19.9: a programmatic set is a projection, never user intent). ``apply_field`` and this row's
        own display swaps block signals, so they never arrive here."""
        if self.editor.hasFocus() or self._editor_dirty:
            return
        self._model_value = text
        tip = text if text and not self._reason else _field_tooltip(self._path, self._reason)
        for widget in (self, self.editor, self.label):
            if widget.toolTip() != tip:
                widget.setToolTip(tip)
        # the row as the context: the collapse is dropped if the row is destroyed first
        QtCore.QTimer.singleShot(0, self, self._collapse_path_editor)

    def _collapse_path_editor(self) -> None:
        if not self._file_name_only or self.editor.hasFocus():
            return
        blocked = self.editor.blockSignals(True)
        try:
            self.editor.setText(self._line_display_value(self._model_value))
            self.editor.setCursorPosition(0)
        finally:
            self.editor.blockSignals(blocked)

    def _draft_edited(self, text: str) -> None:
        self._editor_dirty = True
        self.draftChanged.emit(self._path, text)

    def _emit_editor_edit(self) -> None:
        value = self.current_value()
        if self._file_name_only:
            # A focused Return commit can precede any owner reconciliation,
            # while a focus-out commit must collapse immediately afterwards.
            # Promote the exact edited value locally so neither path can fall
            # back to the old model value.  The normal projection may still
            # reconcile an authoritative replacement once focus is released.
            self._model_value = "" if value is None else str(value)
            self._editor_dirty = False
            if self._model_value:
                for widget in (self, self.editor, self.label):
                    widget.setToolTip(self._model_value)
        self._emit_edit(value)

    def _emit_edit(self, value) -> None:
        edit = ControlFormEdit(path=self._path, value=value)
        self.valueChanged.emit(edit.path, edit.value)


class CoordinateRangeControl(QtWidgets.QWidget):
    """Int range editors and canonical two-handle slider for a display coordinate."""
    rangeChanged = QtCore.Signal(float, float)

    def __init__(self, label, parent=None):
        super().__init__(parent)
        from xdart.gui.widgets.intensity_controls import IntensityRangeSlider
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.row = RangeRow(label=label, display_decimals=3,
            low={"path": ("display_range", "low"), "value": ""},
            high={"path": ("display_range", "high"), "value": ""})
        self.label, self.editors = self.row.label, self.row.editors
        self.label.setMinimumWidth(24)
        self.slider = IntensityRangeSlider(self, range_label=label, continuous=False,
            range_hint="Release a handle to choose the included bin centres. This changes display projections only; saved science is unchanged.")
        layout.addWidget(self.row)
        layout.addWidget(self.slider)
        self._bounds = None
        self.row.valueChanged.connect(self._edited)
        self.slider.sigRangeChanged.connect(self._slid)

    def reconcile(self, label, axis, bounds=None):
        self.label.setText(label)
        self.slider._range_label = label
        low, high = float(axis[0]), float(axis[-1])
        bounds = (low, high) if bounds is None else bounds
        self._bounds = tuple(float(x) for x in bounds)
        self.slider.setDomain(low, high, lower=bounds[0], upper=bounds[1])
        self._sync_editors()
        tip = (f"{label} bin centres included in all displayed maps and curves. "
               "Edit bounds or drag the handles; saved science is unchanged.")
        for widget in (self, self.label, self.row, *self.editors):
            widget.setToolTip(tip)

    def _sync_editors(self):
        for editor, path, value in zip(self.editors, (self.row._low_path, self.row._high_path), self._bounds, strict=True):
            self.row._model_text_by_path[path] = str(value)
            editor.setText(format(value, '.3f'))
            editor.setModified(False)

    def _edited(self, path, value):
        try:
            low, high = self._bounds
            if path[-1] == "low":
                low = float(value)
            else:
                high = float(value)
            from math import isfinite
            if not isfinite(low) or not isfinite(high) or low > high:
                raise ValueError
        except ValueError:
            self._sync_editors()
            return
        self._bounds = low, high
        self.slider.setValues(low, high, emit=False)
        self.rangeChanged.emit(low, high)

    def _slid(self, low, high):
        self._bounds = low, high
        self._sync_editors()
        self.rangeChanged.emit(low, high)


class RangeRow(QtWidgets.QWidget):
    """A compact range row: ``[label] [low] – [high] [✦]`` on one line.

    Coalesces a low/high pair (and an optional auto/enable toggle) that the
    keystone still emits as separate fields, so the panel matches the mockup
    without changing the field model or the write-through: each editor emits
    ``valueChanged(path, value)`` for the SAME path the individual rows used.
    """

    valueChanged = QtCore.Signal(object, object)
    #: A still-uncommitted USER draft on a range bound (line-editor ``textEdited``
    #: — never programmatic ``setText``), revisioned at action time (§12.4).
    draftChanged = QtCore.Signal(object, object)

    def __init__(
        self,
        *,
        label,
        low,
        high,
        toggle=None,
        display_decimals: int | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("controlsRangeRow")
        self._entries: list[tuple[tuple[str, ...], str]] = []
        self._display_decimals = display_decimals
        self._model_text_by_path: dict[tuple[str, ...], str] = {}
        self._display_label = str(label)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(PROCESSING_ROW_GAP)

        self.label = QtWidgets.QLabel(self._display_label)
        self.label.setObjectName("controlsFieldLabel")
        self.label.setMinimumWidth(PROCESSING_ROW_LABEL_WIDTH)
        lay.addWidget(self.label)

        self._low, self._low_path = self._edit(low, lay)
        dash = QtWidgets.QLabel("–")
        dash.setObjectName("controlsRangeDash")
        dash.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(dash)
        self._high, self._high_path = self._edit(high, lay)

        self._toggle = None
        self._toggle_inverts_model = False
        if toggle is not None:
            btn = QtWidgets.QToolButton()
            btn.setObjectName("controlsAutoButton")
            btn.setText("✦")
            btn.setCheckable(True)
            tpath = tuple(toggle["path"])
            # Integration stores whether a range is automatic.  The operator
            # surface speaks the opposite, more useful fact: a lit button
            # means the adjacent entered bounds are active.  Translate only
            # at this presentation boundary; Threshold's direct enable fact
            # and every persisted/science value retain their native polarity.
            self._toggle_inverts_model = bool(
                tpath and tpath[-1].endswith("_auto")
            )
            checked = bool(toggle.get("value"))
            if self._toggle_inverts_model:
                checked = not checked
            btn.setChecked(checked)
            btn.setEnabled(bool(toggle.get("enabled", True)))
            btn.setToolTip(toggle.get(
                "tooltip",
                "Use the entered range values. When off, use Auto."
                if self._toggle_inverts_model else "Enable",
            ))
            btn.setMinimumWidth(31)
            btn.toggled.connect(
                lambda checked, p=tpath: self.valueChanged.emit(
                    p,
                    not bool(checked)
                    if self._toggle_inverts_model else bool(checked),
                )
            )
            lay.addWidget(btn)
            self._toggle = (tpath, btn)

    def _edit(self, spec, lay):
        value = spec.get("value")
        path = tuple(spec["path"])
        self._model_text_by_path[path] = self._model_text(value)
        edit = QtWidgets.QLineEdit(self._display_text(value))
        edit.setModified(False)
        edit.setObjectName("controlsLineEdit")
        edit.setEnabled(bool(spec.get("enabled", True)))
        # Same tooltip rule as _apply_edit: a full render must not leave the
        # bounds without their hover text until the first state UPDATE.
        edit.setToolTip(_field_tooltip(path, spec.get("reason", "")))
        if spec.get("placeholder") is not None:
            edit.setPlaceholderText(spec["placeholder"])
        edit.editingFinished.connect(
            lambda e=edit, p=path: self.valueChanged.emit(
                p, self._current_edit_value(e, p)
            )
        )
        edit.textEdited.connect(
            lambda text, e=edit, p=path: self._draft_edited(e, p, text)
        )
        lay.addWidget(edit, 1)
        return edit, path

    @property
    def editors(self) -> tuple[QtWidgets.QLineEdit, QtWidgets.QLineEdit]:
        """The low and high editors, for a window that drives the row itself (a fit range dragged on its plot)."""
        return self._low, self._high

    def add_trailing_widget(self, widget: QtWidgets.QWidget, stretch: int = 0) -> None:
        """Append a widget to the right of the row (a fit range's Full button), as ``FormRow`` does."""
        self.layout().addWidget(widget, stretch)

    @staticmethod
    def _model_text(value) -> str:
        return "" if value is None else str(value)

    def _display_text(self, value) -> str:
        text = self._model_text(value)
        if self._display_decimals is None or not text:
            return text
        try:
            numeric = float(value)
            if self._display_decimals == 0 and not numeric.is_integer():
                return text
            return format(numeric, f".{self._display_decimals}f")
        except (TypeError, ValueError):
            return text

    def _draft_edited(
        self,
        edit: QtWidgets.QLineEdit,
        path: tuple[str, ...],
        text: str,
    ) -> None:
        edit.setModified(True)
        self.draftChanged.emit(path, text)

    def _current_edit_value(
        self,
        edit: QtWidgets.QLineEdit,
        path: tuple[str, ...],
    ) -> str:
        # A formatted projection is not user intent.  Until textEdited marks a
        # real draft, harvest the exact unformatted model value so focus/blur
        # cannot round a threshold behind the operator's back.
        return (
            edit.text()
            if edit.isModified()
            else self._model_text_by_path[path]
        )

    def current_edits(self) -> tuple[tuple[tuple[str, ...], object], ...]:
        out = []
        if self._toggle is not None:
            checked = bool(self._toggle[1].isChecked())
            out.append((
                self._toggle[0],
                not checked if self._toggle_inverts_model else checked,
            ))
        if self._low.isEnabled():
            out.append((
                tuple(self._low_path),
                self._current_edit_value(self._low, self._low_path),
            ))
        if self._high.isEnabled():
            out.append((
                tuple(self._high_path),
                self._current_edit_value(self._high, self._high_path),
            ))
        return tuple(out)

    def apply_fields(
        self,
        fields_by_path: dict[tuple[str, ...], ControlFormField],
    ) -> bool:
        low = fields_by_path.get(tuple(self._low_path))
        high = fields_by_path.get(tuple(self._high_path))
        if low is None or high is None:
            return False
        self._apply_edit(self._low, low)
        self._apply_edit(self._high, high)
        if self._toggle is not None:
            path, btn = self._toggle
            field = fields_by_path.get(tuple(path))
            if field is None:
                return False
            was_blocked = btn.blockSignals(True)
            try:
                checked = bool(field.value)
                if self._toggle_inverts_model:
                    checked = not checked
                if btn.isChecked() != checked:
                    btn.setChecked(checked)
            finally:
                btn.blockSignals(was_blocked)
            btn.setEnabled(bool(field.enabled))
            fallback = (
                "Use the entered range values. When off, use Auto."
                if self._toggle_inverts_model else "Enable this range."
            )
            btn.setToolTip(
                _field_tooltip(field.path, field.reason) or fallback
            )
        return True

    def _apply_edit(
        self,
        edit: QtWidgets.QLineEdit,
        field: ControlFormField,
    ) -> None:
        edit.setEnabled(bool(field.enabled))
        edit.setToolTip(_field_tooltip(field.path, field.reason))
        if field.placeholder is not None:
            edit.setPlaceholderText(field.placeholder)
        path = tuple(field.path)
        self._model_text_by_path[path] = self._model_text(field.value)
        value = self._display_text(field.value)
        if edit.hasFocus():
            return
        if edit.text() != value:
            was_blocked = edit.blockSignals(True)
            try:
                edit.setText(value)
            finally:
                edit.blockSignals(was_blocked)
        # Even when two exact model values share one rounded presentation,
        # this accepted unfocused projection supersedes the prior UI draft.
        edit.setModified(False)


class PillRow(QtWidgets.QWidget):
    """A left-aligned row of compact toggle *pills* (the bool toggles).

    Replaces the full-width stacked toggle buttons with mockup-style pills:
    several share a line, each sized to its label.  Each pill emits
    ``valueChanged(path, bool)`` so the write-through is unchanged.
    """

    valueChanged = QtCore.Signal(object, object)

    def __init__(
        self,
        fields: Sequence[ControlFormField],
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("controlsPillRow")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._pills: list[tuple[tuple[str, ...], QtWidgets.QPushButton]] = []
        for field in fields:
            btn = QtWidgets.QPushButton(field.label)
            # Reuse the accent-when-checked toggle styling, but content-sized and
            # fully rounded, via a DEDICATED object name (grouped with
            # controlsToggleButton in the theme).  The fully-rounded radius
            # also needs the button tall enough for macOS to draw the rounded
            # bezel -- the theme floors it with a min-height (see the
            # #controlsPillButton rule).
            btn.setObjectName("controlsPillButton")
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
        return tuple(
            (p, bool(btn.isChecked()))
            for p, btn in self._pills
        )

    def apply_fields(
        self,
        fields_by_path: dict[tuple[str, ...], ControlFormField],
    ) -> bool:
        for path, btn in self._pills:
            field = fields_by_path.get(tuple(path))
            if field is None:
                return False
            was_blocked = btn.blockSignals(True)
            try:
                checked = bool(field.value)
                if btn.isChecked() != checked:
                    btn.setChecked(checked)
            finally:
                btn.blockSignals(was_blocked)
            btn.setEnabled(bool(field.enabled))
            btn.setToolTip(_field_tooltip(field.path, field.reason))
        return True


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
        self.setObjectName("controlsSegmentedControl")
        self._path = tuple(path)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        self._segments: list[tuple[object, QtWidgets.QPushButton]] = []
        for idx, (label, opt_value) in enumerate(options):
            btn = QtWidgets.QPushButton(label)
            btn.setObjectName("controlsSegmentButton")
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

    def apply_field(self, field: ControlFormField) -> bool:
        if tuple(field.path) != self._path:
            return False
        for opt_value, btn in self._segments:
            btn.setEnabled(bool(field.enabled))
            btn.setToolTip(_field_tooltip(field.path, field.reason))
            was_blocked = btn.blockSignals(True)
            try:
                checked = opt_value == field.value
                if btn.isChecked() != checked:
                    btn.setChecked(checked)
            finally:
                btn.blockSignals(was_blocked)
        return True


class ControlsPanel(QtWidgets.QWidget):
    """Render one immutable :class:`ControlsProjection`.

    It emits intents only. The owning page builds state and decides how to
    handle actions and field edits.
    """

    controlActionRequested = QtCore.Signal(object)
    fieldValueChanged = QtCore.Signal(object, object)
    #: A focused, still-uncommitted line-editor draft (§12.4) — revisioned at
    #: action time, applied only when it commits via fieldValueChanged.
    fieldDraftChanged = QtCore.Signal(object, object)
    fieldBrowseRequested = QtCore.Signal(object)

    def __init__(
        self,
        parent=None,
        *,
        experiment_title: str = "Sample & measurement",
        show_section_numbers: bool = True,
    ):
        super().__init__(parent)
        self._experiment_title = experiment_title
        self._show_section_numbers = bool(show_section_numbers)
        self.setObjectName("controlsPanel")
        # LV-UI-7: no explicit root minimum — the layout's minimumSizeHint is
        # the panel's one true width floor, so embedding surfaces (the vNext
        # shell's width-hugging scroll column) can derive exactly what the
        # panel needs and the boxes contract instead of clipping.  The old
        # hand-pinned 360 predated the row-based redesign and forced a 61 px
        # horizontal overflow at 1440x900.
        lay = QtWidgets.QVBoxLayout(self)
        self._root_layout = lay
        lay.setContentsMargins(5, 5, 5, 5)
        # Roomier gap between the workflow sections for visual separation (Vivek).
        lay.setSpacing(12)

        self.project_card = self._make_section(SectionId.PROJECT)
        self.source_card = self._make_section(SectionId.SOURCE)
        self.experiment_card = self._make_section(SectionId.EXPERIMENT)
        self.processing_card = self._make_section(SectionId.PROCESSING)
        lay.addWidget(self.project_card)
        lay.addWidget(self.experiment_card)
        lay.addWidget(self.source_card)
        lay.addWidget(self.processing_card)
        lay.addStretch(1)
        self._projection: ControlsProjection | None = None
        self._apply_spacing(current_spacing_tokens())

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in {
            QtCore.QEvent.Type.StyleChange,
            QtCore.QEvent.Type.FontChange,
        } and hasattr(self, "_root_layout"):
            self._apply_spacing(current_spacing_tokens())

    def _apply_spacing(self, tokens: SpacingTokens) -> None:
        margin = _spaced(5, tokens)
        self._root_layout.setContentsMargins(
            margin, margin, margin, margin
        )
        self._root_layout.setSpacing(tokens.tools_vertical_margin)

    def _make_section(self, section: SectionId) -> SectionCard:
        number, title, accent = _SECTION_META[section]
        if not self._show_section_numbers:
            number = ""
        return SectionCard(title, number=number, accent=accent)

    def set_processing_widget(
        self,
        widget: QtWidgets.QWidget | None,
        *,
        visible: bool = True,
    ) -> None:
        self.processing_card.set_embedded_widget(widget, visible=visible)

    def set_source_widget(
        self,
        widget: QtWidgets.QWidget | None,
        *,
        visible: bool = True,
    ) -> None:
        """Mount the shared authoritative source observer in the Source card."""
        self.source_card.set_embedded_widget(widget, visible=visible)

    def source_widget_visible(self) -> bool:
        """The Source card's EXPLICIT embedded-source visibility (§21.5 req 2).

        The reviewed accessor the Controls transaction's source-recovery receipt
        proves against — what the projection last applied, never the desired
        `config is not None`."""
        return self.source_card.embedded_visible()

    @property
    def projection(self) -> ControlsProjection | None:
        return self._projection

    def reconcile(self, state: ControlsProjection) -> bool:
        """Render one projection, retaining rows when its schema is unchanged.

        Returns ``True`` when the in-place path retained the mounted widgets,
        otherwise ``False`` after a required schema rebuild.
        """

        if type(state) is not ControlsProjection:
            raise TypeError("ControlsPanel requires an exact ControlsProjection")
        if self._apply_projection_update(state):
            return True
        self._projection = state
        self._render_projection(state)
        return False

    def _apply_projection_update(self, state: ControlsProjection) -> bool:
        """Apply a same-schema snapshot without tearing down row widgets."""

        if self._projection is None:
            return False
        current_sequence = tuple(self._projection.fields)
        next_sequence = tuple(state.fields)
        current_fields = {
            tuple(field.path): field
            for field in current_sequence
        }
        next_fields = {
            tuple(field.path): field
            for field in next_sequence
        }
        # The row walkers below prove that every existing row still has an
        # owner.  Exact set equality also proves the inverse: no newly
        # projected field is silently omitted by a stale fast-path layout —
        # LV-UI-5: switching Standard→Grazing ADDS the θ-motor field, and the
        # in-place path must force a rebuild for it to appear.  (Unlike the
        # E4-era guard, a changed combo CHOICE inventory does NOT force a
        # rebuild: PM1's accepted apply_field reconciles vocabulary in place.)
        def render_schema(fields):
            return tuple(
                (
                    field.section,
                    field.label,
                    tuple(field.path),
                    field.kind,
                    bool(field.browse),
                    field.parameter_group,
                )
                for field in fields
            )

        if render_schema(current_sequence) != render_schema(next_sequence):
            return False
        next_experiment = tuple(
            field for field in next_sequence
            if field.section is SectionId.EXPERIMENT
        )

        def processing_groups(fields):
            grouped: dict[str, list[ControlFormField]] = {}
            for field in fields:
                if (field.section is not SectionId.PROCESSING
                        or field.parameter_group):
                    continue
                group = self._processing_group_for_path(field.path)
                grouped.setdefault(group, []).append(field)
            return grouped
        current_action_schema = tuple(
            (
                section,
                len(actions),
                tuple(spec.action for spec in actions),
            )
            for section, actions in self._projection.section_actions.items()
        )
        next_action_schema = tuple(
            (
                section,
                len(actions),
                tuple(spec.action for spec in actions),
            )
            for section, actions in state.section_actions.items()
        )
        if current_action_schema != next_action_schema:
            return False

        mounted_specs = (
            tuple(self._experiment_producers(state))
            if state.fields_for(SectionId.EXPERIMENT)
            else ()
        ) + state.actions_for(SectionId.PROCESSING)
        expected_actions: dict[
            tuple[SectionId, ControlAction], ControlActionSpec
        ] = {}
        for spec in mounted_specs:
            key = (spec.section, spec.action)
            if key in expected_actions:
                return False
            expected_actions[key] = spec
        mounted_actions: dict[
            tuple[SectionId, ControlAction], ActionButton
        ] = {}
        for button in self.findChildren(ActionButton):
            key = (button.spec.section, button.spec.action)
            if key in mounted_actions:
                return False
            mounted_actions[key] = button
        if mounted_actions.keys() != expected_actions.keys():
            return False

        # Compact popup launchers are not FormRows.  Validate their ownership
        # before any in-place mutation, then refresh the field snapshots they
        # use when opened.  This keeps the controls-only metadata refresh fast
        # without leaving an enabled GI/energy popup backed by stale fields.
        source_field = next_fields.get(_SOURCE_ENERGY_PATH)
        source_more = [
            button
            for button in self.source_card.body.findChildren(
                QtWidgets.QToolButton
            )
            if button.objectName() == "controlsMoreButton"
            and button.property("role") == "sourceEnergy"
        ]
        if (source_field is None and source_more) or (
            source_field is not None and len(source_more) != 1
        ):
            return False

        gi_popup_paths = (
            ("GI", "sample_orientation"),
            ("GI", "tilt_angle"),
        )
        gi_popup_fields = tuple(
            next_fields[path]
            for path in gi_popup_paths
            if path in next_fields
        )
        gi_more = [
            button
            for button in self.experiment_card.body.findChildren(
                QtWidgets.QToolButton
            )
            if button.objectName() == "controlsMoreButton"
            and button.property("role") != "sourceEnergy"
        ]
        if (not gi_popup_fields and gi_more) or (
            gi_popup_fields and len(gi_more) != 1
        ):
            return False

        detector_paths = {
            ("Signal", "poni_file"),
            ("Calibration", "poni_file"),
            ("Signal", "mask_file"),
        }
        gi_paths = {
            ("GI", "Grazing"),
            ("GI", "th_motor"),
            ("GI", "th_val"),
            *gi_popup_paths,
        }
        experiment_statuses = {}
        if any(field.path in detector_paths for field in next_experiment):
            experiment_statuses["Detector"] = self._detector_status(
                next_experiment, state.detector_summary,
            )
        if any(field.path in gi_paths for field in next_experiment):
            experiment_statuses[self._experiment_title] = (
                self._experiment_status(next_experiment)
            )
        experiment_subsections = {
            subsection.title.text(): subsection
            for subsection in self.experiment_card.body.findChildren(
                SubsectionCard
            )
        }
        if experiment_subsections.keys() != experiment_statuses.keys():
            return False

        next_processing_groups = processing_groups(next_sequence)
        processing_statuses = {
            name: self._group_status(next_processing_groups[name])
            for name in ("1-D", "2-D", "Conditioning", "Background")
            if next_processing_groups.get(name)
        }
        processing_subsections = {
            subsection.title.text(): subsection
            for subsection in self.processing_card.body.findChildren(
                SubsectionCard
            )
        }
        if processing_subsections.keys() != processing_statuses.keys():
            return False

        self._projection = state
        fields_by_path = {
            tuple(field.path): field
            for field in state.fields
        }

        updated = 0
        for row in self.findChildren(FormRow):
            field = fields_by_path.get(tuple(row.path))
            if field is None:
                return False
            if not row.apply_field(field):
                return False
            updated += 1
        for row in self.findChildren(RangeRow):
            if not row.apply_fields(fields_by_path):
                return False
            updated += 1
        for row in self.findChildren(PillRow):
            if not row.apply_fields(fields_by_path):
                return False
            updated += 1
        for row in self.findChildren(SegmentedControl):
            field = fields_by_path.get(tuple(row.path))
            if field is None:
                return False
            if not row.apply_field(field):
                return False
            updated += 1

        for key, spec in expected_actions.items():
            mounted_actions[key].apply_spec(spec)

        if source_field is not None:
            source_button = source_more[0]
            source_button._controls_field = source_field
            source_button.setEnabled(bool(source_field.enabled))
            source_button.setToolTip(
                _field_tooltip(source_field.path, source_field.reason)
                or "Energy source"
            )
            if not source_field.enabled:
                self._close_source_energy_popup()
        if gi_popup_fields:
            gi_button = gi_more[0]
            gi_button._controls_fields = gi_popup_fields
            gi_button.setEnabled(
                any(field.enabled for field in gi_popup_fields)
            )
            gi_button.setToolTip(
                next(
                    (
                        _field_tooltip(field.path, field.reason)
                        for field in gi_popup_fields
                        if field.reason
                    ),
                    "More GI options: Orientation, Tilt Angle",
                )
            )
            if not any(field.enabled for field in gi_popup_fields):
                self._close_gi_more_popup()

        for title, status in experiment_statuses.items():
            label = experiment_subsections[title].status
            label.setText(status)
            label.setVisible(bool(status))
        for title, status in processing_statuses.items():
            label = processing_subsections[title].status
            label.setText(status)
            label.setVisible(bool(status))

        if fields_by_path and not updated:
            return False
        self._apply_card_visibility(state)
        return True

    def focused_form_edit(self) -> ControlFormEdit | None:
        """The single bound line editor the user is CURRENTLY editing (has
        keyboard focus), as a ``ControlFormEdit``, or ``None`` when none is
        focused.

        §13.11 owner 5: the edit journal already holds every keystroke draft
        (FormRow + RangeRow ``textEdited`` → ``draftChanged``); the ONLY value not
        yet revisioned is the one the user is mid-editing, whose final text may
        post-date its last draft signal.  The transaction flushes THIS editor
        alone (exactly one revision) instead of importing the full visible-form
        snapshot."""
        for row in self.findChildren(FormRow):
            editor = getattr(row, "editor", None)
            if (editor is not None and hasattr(editor, "hasFocus")
                    and editor.hasFocus() and hasattr(editor, "text")):
                return ControlFormEdit(path=row.path, value=row.current_value())
        for row in self.findChildren(RangeRow):
            for editor, path in (
                (row._low, row._low_path), (row._high, row._high_path)
            ):
                if (editor is not None and editor.hasFocus()):
                    return ControlFormEdit(
                        path=path,
                        value=row._current_edit_value(editor, path),
                    )
        return None

    def _render_projection(self, state: ControlsProjection) -> None:
        # Popup rows carry captured field snapshots. A schema rebuild replaces
        # those owners, so dispose the popup subtree before mounting new rows.
        self._close_gi_more_popup()
        self._close_source_energy_popup()
        cards = (
            self.project_card,
            self.source_card,
            self.experiment_card,
            self.processing_card,
        )
        for card in cards:
            card.clear_rows()

        self._render_plain_bound_section(
            self.project_card,
            state.fields_for(SectionId.PROJECT),
        )
        self._render_source_bound_section(
            state.fields_for(SectionId.SOURCE),
        )
        self._render_experiment_bound_section(
            state,
            state.fields_for(SectionId.EXPERIMENT),
        )
        self._render_processing_bound_section(
            state,
            state.fields_for(SectionId.PROCESSING),
        )
        self._apply_card_visibility(state)

    def _apply_card_visibility(self, state: ControlsProjection) -> None:
        self.apply_card_visibility(
            state.value_for(_PROJECT_ROOT_PATH, None),
            viewer_mode=state.processing_page is ProcessingPage.VIEWER,
            experiment=bool(state.fields_for(SectionId.EXPERIMENT)),
        )

    def apply_card_visibility(
        self,
        project_root: object,
        *,
        viewer_mode: bool = False,
        experiment: bool = True,
    ) -> None:
        """The section gate: Source, Experiment and Processing show once a
        Project folder is named (None: no Project field), never in a viewer.
        Public for another page's own Project value (Stitch)."""
        project_ready = (
            project_root is None
            or type(project_root) is str and bool(project_root.strip())
        )
        self.project_card.show()
        self.source_card.setVisible(not viewer_mode and project_ready)
        self.experiment_card.setVisible(experiment and not viewer_mode and project_ready)
        self.processing_card.setVisible(not viewer_mode and project_ready)

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
        self.render_source_rows(self.source_card, fields)

    def render_source_rows(
        self,
        card: SectionCard,
        fields: Iterable[ControlFormField],
    ) -> dict[tuple[str, ...], FormRow]:
        """SOURCE layout with a few coalesced rows (Vivek): the mode combo + the
        directory-only Subdirs toggle share a row, and File Type + Meta Type share
        a row.  Everything else renders one field per row, in field order.

        Public for another page's own source values (Stitch): returns each field's
        row by path. Edits emit ``fieldValueChanged`` with the fields' paths."""
        fields = tuple(fields)
        by_path = {field.path: field for field in fields}
        rendered: set[tuple[str, ...]] = set()
        built: dict[tuple[str, ...], FormRow] = {}

        def row_for(
                *paths: tuple[str, ...],
                stretches=None,
                tight=(),
                energy_field: ControlFormField | None = None,
        ) -> None:
            present = [by_path[p] for p in paths
                       if p in by_path and p not in rendered]
            if energy_field is not None and energy_field.path not in rendered:
                rendered.add(energy_field.path)
            if not present:
                if energy_field is not None:
                    container = QtWidgets.QWidget()
                    lay = QtWidgets.QHBoxLayout(container)
                    lay.setContentsMargins(0, 0, 0, 0)
                    lay.setSpacing(8)
                    lay.addStretch(1)
                    lay.addWidget(self._make_source_energy_button(energy_field))
                    card.add_row(container)
                return
            rendered.update(field.path for field in present)
            if len(present) == 1 and not tight and energy_field is None:
                built[present[0].path] = row = self._make_bound_row(present[0])
                card.add_row(row)
                return
            container = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(container)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(8)
            for field in present:
                built[field.path] = sub = self._make_bound_row(field)
                if field.path in tight:
                    # Paired labels use their text width instead of the full
                    # form's aligned label column.
                    if getattr(sub, "label", None) is not None:
                        sub.label.setMinimumWidth(0)
                        sub.label.setSizePolicy(
                            QtWidgets.QSizePolicy.Maximum,
                            QtWidgets.QSizePolicy.Preferred,
                        )
                    if sub.layout() is not None:
                        sub.layout().setSpacing(6)
                is_bool = getattr(field.kind, "value", field.kind) == "bool"
                st = 0 if is_bool else 1
                if stretches and field.path in stretches:
                    st = stretches[field.path]
                lay.addWidget(sub, st)
            if energy_field is not None:
                lay.addWidget(self._make_source_energy_button(energy_field), 0)
            card.add_row(container)

        # Mode combo + (directory-only) Subdirs toggle on one row.
        row_for(("Signal", "inp_type"), ("Signal", "include_subdir"))
        row_for(("Signal", "img_dir"))                           # Directory
        row_for(("Signal", "File"))                              # Image File
        # Directory mode: File Type | Meta Type share a row.  Other source modes
        # keep Meta Type on the standard label column so it aligns with Source /
        # Image File above it.
        energy_field = by_path.get(_SOURCE_ENERGY_PATH)
        pair_meta_type = ("Signal", "img_ext") in by_path
        row_for(("Signal", "img_ext"), ("Signal", "meta_ext"),
                stretches={("Signal", "img_ext"): 9, ("Signal", "meta_ext"): 7},
                tight={("Signal", "meta_ext")} if pair_meta_type else (),
                energy_field=energy_field)
        row_for(("Signal", "Filter"))                            # Filter
        row_for(("Signal", "meta_dir"))                          # SPEC Dir
        # Anything else (e.g. NeXus File / Entry) one per row, in order.
        for field in fields:
            if field.path not in rendered:
                built[field.path] = row = self._make_bound_row(field)
                card.add_row(row)
        return built

    def _make_source_energy_button(
            self, field: ControlFormField) -> QtWidgets.QToolButton:
        btn = QtWidgets.QToolButton()
        btn.setText("…")
        btn.setObjectName("controlsMoreButton")
        btn.setProperty("role", "sourceEnergy")
        btn.setToolTip("Energy source")
        btn.setEnabled(bool(field.enabled))
        btn._controls_field = field
        btn.clicked.connect(
            lambda _=False, b=btn: self._open_source_energy_popup(
                b._controls_field
            )
        )
        return btn

    def _close_source_energy_popup(self) -> None:
        popup = getattr(self, "_source_energy_popup", None)
        if popup is not None:
            popup.close()
            popup.setParent(None)
            popup.deleteLater()
        self._source_energy_popup = None

    def _open_source_energy_popup(self, field: ControlFormField) -> None:
        self._close_source_energy_popup()
        popup = QtWidgets.QWidget(self, QtCore.Qt.Tool)
        popup.setObjectName("controlsEnergyPopup")
        popup.setWindowTitle("Energy Source")
        lay = QtWidgets.QVBoxLayout(popup)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        value = str(field.value or "poni").strip().lower()
        seg = SegmentedControl(
            field.path,
            _SOURCE_ENERGY_OPTIONS,
            value=value if value in {"poni", "metadata"} else "poni",
            enabled=bool(field.enabled),
            reason=field.reason or "",
        )
        seg.valueChanged.connect(self.fieldValueChanged)
        lay.addWidget(seg)
        self._source_energy_popup = popup
        popup.show()
        popup.raise_()
        popup.activateWindow()

    def _render_experiment_bound_section(
        self,
        state: ControlsProjection,
        fields: tuple[ControlFormField, ...],
    ) -> None:
        if not fields:
            return
        self.render_experiment_rows(
            self.experiment_card, fields, self._experiment_producers(state),
            detector_summary=state.detector_summary,
        )

    def render_experiment_rows(
        self,
        card: SectionCard,
        fields: tuple[ControlFormField, ...],
        producers: Sequence[ControlActionSpec] = (),
        *,
        detector_summary: str = "",
    ) -> dict[object, QtWidgets.QWidget]:
        """EXPERIMENT layout: the producer actions on top, then the Detector
        subsection (calibration and mask) and the Sample & measurement one (GI).

        Public for another page's own values (Stitch): returns each Detector
        field's row by path and each producer's button by action. Edits emit
        ``fieldValueChanged``; a button ``controlActionRequested``."""
        built: dict[object, QtWidgets.QWidget] = {}
        # Producer actions (Calibrate / Refine / Make Mask) sit at the top of the
        # section they write into — co-located with §3 state, not a top bar.
        if producers:
            row = QtWidgets.QWidget()
            row.setObjectName("controlsActionRow")
            prow = QtWidgets.QHBoxLayout(row)
            prow.setContentsMargins(0, 0, 0, 3)
            prow.setSpacing(5)
            for spec in producers:
                built[spec.action] = btn = ActionButton(spec)
                btn.actionRequested.connect(self.controlActionRequested)
                prow.addWidget(btn, 1)
            card.add_row(row)
        detector_paths = {
            ("Signal", "poni_file"),
            ("Calibration", "poni_file"),
            ("Signal", "mask_file"),
        }
        # Sample & measurement shows a Standard | Grazing segmented control (not a
        # group-header checkbox — that was the #56 repaint class).  The GI detail
        # fields (motor, value, orientation, tilt) render inline beneath it, but
        # only in Grazing mode: the Controls projection gates their PRESENCE on
        # the Grazing state (progressive disclosure), so here they are simply
        # rendered when present.
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
                self._detector_status(fields, detector_summary),
            ),
            (
                self._experiment_title,
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
            if title == self._experiment_title:
                self._render_sample_measurement_rows(group, group_fields)
            else:
                for field in group_fields:
                    built[field.path] = row = self._make_bound_row(field)
                    group.add_row(row)
            card.add_row(group)
        return built

    @property
    def experiment_title(self) -> str:
        """The Sample & measurement subsection's title, for another page's GI rows."""
        return self._experiment_title

    def render_gi_rows(
        self,
        group: "SubsectionCard",
        fields: tuple[ControlFormField, ...],
    ) -> None:
        """Int's grazing-incidence rows for another page's own GI values (Stitch): the
        Standard|Grazing control, the incidence motor and Manual angle, and '…' for the
        orientation and tilt. Edits emit ``fieldValueChanged`` with Int's GI paths."""
        self._render_sample_measurement_rows(group, fields)

    def _render_sample_measurement_rows(
        self,
        group: "SubsectionCard",
        fields: tuple[ControlFormField, ...],
    ) -> None:
        """Render the Standard|Grazing segmented control, then a compact GI row:
        the θ motor combo, the manual θ value (manual mode only), and a '…'
        button that opens a small popup with the less-used Orientation + Tilt
        Angle options.  The Controls projection drops detail fields in Standard
        mode."""
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
            more.setObjectName("controlsMoreButton")
            more.setToolTip("More GI options: Orientation, Tilt Angle")
            more.setEnabled(any(field.enabled for field in popup_fields))
            more._controls_fields = tuple(popup_fields)
            more.clicked.connect(
                lambda _=False, b=more: self._open_gi_more_popup(
                    b._controls_fields
                )
            )
            lay.addWidget(more, 0)
        group.add_row(row)

    def _close_gi_more_popup(self) -> None:
        """Close + dispose the GI '…' options popup if open, clearing the ref so a
        stale popup can't be harvested by any form reader (F1/F2)."""
        popup = getattr(self, "_gi_options_popup", None)
        if popup is not None:
            popup.close()
            # deleteLater() is async — its FormRow children would still be found
            # by findChildren() until the event loop runs.
            # setParent(None) detaches the subtree from this panel NOW, so a
            # rebuild can't harvest its stale rows before the deferred delete.
            popup.setParent(None)
            popup.deleteLater()
        self._gi_options_popup = None

    def _open_gi_more_popup(self, fields: tuple[ControlFormField, ...]) -> None:
        """Small floating popup for the less-used GI options (Orientation, Tilt
        Angle).  Its rows write through like any other control row."""
        self._close_gi_more_popup()
        popup = QtWidgets.QWidget(self, QtCore.Qt.Tool)
        popup.setObjectName("controlsGIMorePopup")
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
        state: ControlsProjection,
        fields: tuple[ControlFormField, ...],
    ) -> None:
        # Advanced (parameter_group-backed) fields are NOT rendered inline:
        # the "Advanced" button on the Reintegrate row opens the
        # "Integration — Advanced Settings" dialog, the SINGLE editing surface
        # for the bai_1d/2d parameter trees (maintainer decision 2026-07-12 —
        # the collapsed sub-panel duplicated that dialog field-for-field).
        inline_fields = [field for field in fields if not field.parameter_group]
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

        self._add_actions(
            self.processing_card,
            state.actions_for(SectionId.PROCESSING),
            expand=True,
        )

    def render_processing_rows(
        self,
        sub: SubsectionCard,
        fields: Sequence[ControlFormField],
    ) -> dict[tuple[str, ...], QtWidgets.QWidget]:
        """Int's rows of one Processing group (1-D, 2-D, Conditioning) for another page's own
        values (Stitch): the Axis row with its Pts, a RangeRow per low/high/auto triple and the
        Threshold, pills for toggles. Returns each field's row by path (a range's three paths
        its RangeRow). Edits emit ``fieldValueChanged`` with the fields' paths."""
        return self._render_processing_group_rows(sub, fields)

    def render_count_row(
        self,
        section: SectionCard | SubsectionCard,
        label: str,
        fields: Sequence[ControlFormField],
    ) -> dict[tuple[str, ...], FormRow]:
        """Mount labeled numeric fields on one processing-aligned row.

        Each field's label names its axis; the existing FormRow owns its
        validation, drafts and committed edits.
        """
        container = QtWidgets.QWidget()
        container.setToolTip("Number of points along each labeled axis.")
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(PROCESSING_ROW_GAP)
        heading = QtWidgets.QLabel(label)
        heading.setObjectName("controlsFieldLabel")
        heading.setFixedWidth(PROCESSING_ROW_LABEL_WIDTH)
        heading.setWordWrap(True)
        heading.setToolTip(container.toolTip())
        layout.addWidget(heading)
        built = {}
        for field in fields:
            row = self._make_bound_row(field)
            row.label.setMinimumWidth(0)
            row.layout().setSpacing(PROCESSING_ROW_GAP)
            layout.addWidget(row, 1)
            built[field.path] = row
        section.add_row(container)
        return built

    @staticmethod
    def _align_processing_row(row: FormRow) -> None:
        row.label.setFixedWidth(PROCESSING_ROW_LABEL_WIDTH)
        row.label.setWordWrap(True)
        row.layout().setSpacing(PROCESSING_ROW_GAP)

    def _render_processing_group_rows(
        self,
        sub: SubsectionCard,
        fields: Sequence[ControlFormField],
    ) -> dict[tuple[str, ...], QtWidgets.QWidget]:
        """Render a Processing group, coalescing low/high(/auto) triples and the
        Threshold min/max into one compact :class:`RangeRow` each (mockup)."""
        by_path = {field.path: field for field in fields}
        consumed: set[tuple[str, ...]] = set()
        built: dict[tuple[str, ...], QtWidgets.QWidget] = {}
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
                built.update(dict.fromkeys((field.path for field in pending_pills), row))
                if any(
                    field.path in {
                        ("Signal", "series_average"),
                    }
                    for field in pending_pills
                ):
                    row.layout().setContentsMargins(
                        0, PROCESSING_PILL_TOP_MARGIN, 0, 0
                    )
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
                    row = self._make_range_row(low_f, high_f, by_path.get(auto_p))
                    sub.add_row(row)
                    built.update(dict.fromkeys((p for p in (low_p, high_p, auto_p) if p in by_path), row))
                    consumed.update({low_p, high_p, auto_p})
                    continue
            # A bare _high / _auto whose _low partner is in this group is folded
            # into the RangeRow above; skip it.
            if last.endswith(("_high", "_auto")) and self._range_low_path(path) in by_path:
                consumed.add(path)
                continue
            # Threshold: (Mask, Threshold)=enable + (Mask, min) + (Mask, max).
            # One threshold enable covers the automatic or explicit band.
            if path == ("Mask", "min") and ("Mask", "max") in by_path:
                flush_pills()
                toggle_field = by_path.get(("Mask", "Threshold"))
                row = self._make_range_row(
                    field, by_path[("Mask", "max")], toggle_field,
                    label="Threshold", display_decimals=0)
                sub.add_row(row)
                built.update(dict.fromkeys((p for p in (("Mask", "min"), ("Mask", "max"), ("Mask", "Threshold"))
                                            if p in by_path), row))
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
                # RangeRow is the alignment owner for the bounds directly
                # below.  Match its label width and gap so the Axis combo and
                # both range editors share one exact left edge.
                self._align_processing_row(row)
                if point_fields:
                    built.update(self._attach_points_to_row(row, point_fields))
                    points_attached = True
                sub.add_row(row)
                built[path] = row
                continue
            built[path] = row = self._make_bound_row(field)
            self._align_processing_row(row)
            sub.add_row(row)
        flush_pills()
        # Safety net: never drop the point fields if a group has them but no
        # Axis row to host them.
        if point_fields and not points_attached:
            built.update(self._add_points_header(sub, point_fields))
        return built

    def _add_points_header(
        self,
        sub: SubsectionCard,
        point_fields: Sequence[ControlFormField],
    ) -> dict[tuple[str, ...], FormRow]:
        """Put the point-count field(s) on the subsection header as ``Pts [n] …``.

        Reuses :class:`FormRow` (label hidden) so the editors stay harvestable by
        the draft/focused-edit seam and route through the same write-through path."""
        label = QtWidgets.QLabel("Pts")
        label.setObjectName("controlsHeaderLabel")
        sub.add_header_widget(label)
        built = {}
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
            row.draftChanged.connect(self.fieldDraftChanged)
            sub.add_header_widget(row)
            built[field.path] = row
        return built

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
        display_decimals: int | None = None,
    ) -> "RangeRow":
        if label is None:
            label = low_field.label
            for suffix in (" Low", " Range"):
                if label.endswith(suffix):
                    label = label[: -len(suffix)]
                    break
        toggle = None
        if toggle_field is not None:
            is_auto_fact = bool(
                toggle_field.path
                and toggle_field.path[-1].endswith("_auto")
            )
            toggle = {
                "path": toggle_field.path,
                "value": toggle_field.value,
                "enabled": toggle_field.enabled,
                "tooltip": toggle_field.reason or (
                    "Use the entered range values. When off, use Auto."
                    if is_auto_fact else "Enable this range."
                ),
            }
        row = RangeRow(
            label=label,
            low={"path": low_field.path, "value": low_field.value,
                 "enabled": low_field.enabled, "reason": low_field.reason,
                 "placeholder": low_field.placeholder},
            high={"path": high_field.path, "value": high_field.value,
                  "enabled": high_field.enabled, "reason": high_field.reason,
                  "placeholder": high_field.placeholder},
            toggle=toggle,
            display_decimals=display_decimals,
        )
        row.valueChanged.connect(self.fieldValueChanged)
        row.draftChanged.connect(self.fieldDraftChanged)
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
    def _group_status(fields: Sequence[ControlFormField]) -> str:
        disabled = sum(1 for field in fields if not field.enabled)
        if disabled and disabled == len(fields):
            return "locked"
        return ""

    _PRODUCER_ORDER = {
        ControlAction.CALIBRATE: 0,
        ControlAction.MAKE_MASK: 1,
    }

    def _experiment_producers(self, state: ControlsProjection):
        """The instrument-producer actions, sorted Calibrate · Make Mask."""
        return sorted(
            state.actions_for(SectionId.EXPERIMENT),
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
            minimum=field.minimum,
            maximum=field.maximum,
            allow_blank=field.allow_blank,
        )
        row.valueChanged.connect(self.fieldValueChanged)
        row.draftChanged.connect(self.fieldDraftChanged)
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
    ) -> dict[tuple[str, ...], FormRow]:
        """Park the compact ``Pts [n] …`` cluster on the right of an Axis row.

        Reuses :class:`FormRow` (label hidden) so the editors stay harvestable by
        the draft/focused-edit seam and route through the same write-through path."""
        label = QtWidgets.QLabel("Pts")
        label.setObjectName("controlsHeaderLabel")
        row.add_trailing_widget(label)
        built = {}
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
            pr.draftChanged.connect(self.fieldDraftChanged)
            row.add_trailing_widget(pr)
            built[field.path] = pr
        return built

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

    def render_actions(
        self,
        card: SectionCard,
        actions: tuple[ControlActionSpec, ...],
        *,
        expand: bool = False,
    ) -> dict[object, ActionButton]:
        """A section's action row (Int's Reintegrate · Advanced) for another page's
        own actions (Stitch's Preview · Retry · Advanced): each button by action;
        a click emits ``controlActionRequested``."""
        return self._add_actions(card, actions, expand=expand)

    def set_editors_enabled(self, enabled: bool) -> None:
        """Lock or unlock every row of the four sections except their action rows,
        whose buttons keep their own availability: Int locks its rows field by field
        through its projection; a page that renders its own rows and actions (Stitch,
        RSM) locks them here, not by disabling the panel, whose disabled ancestry
        would also disable an action that must stay usable (a pending result's Retry,
        Codex's RSM stage-1 review F2)."""
        for card in (
            self.project_card,
            self.experiment_card,
            self.source_card,
            self.processing_card,
        ):
            for index in range(card.body_layout.count()):
                row = card.body_layout.itemAt(index).widget()
                if row is not None and row.objectName() != "controlsActionRow":
                    row.setEnabled(bool(enabled))

    def _add_actions(
        self,
        card: SectionCard,
        actions: tuple[ControlActionSpec, ...],
        *,
        expand: bool = False,
    ) -> dict[object, ActionButton]:
        if not actions:
            return {}
        row = QtWidgets.QWidget()
        row.setObjectName("controlsActionRow")
        lay = QtWidgets.QHBoxLayout(row)
        lay.setContentsMargins(0, 2, 0, 0)
        lay.setSpacing(5)
        if not expand:
            lay.addStretch(1)
        built = {}
        for spec in actions:
            built[spec.action] = btn = ActionButton(spec)
            btn.actionRequested.connect(self.controlActionRequested)
            lay.addWidget(btn, 1 if expand else 0)
        card.add_row(row)
        return built
