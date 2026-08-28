"""Additive oracle for the canonical/visual ControlsPanel synthesis."""

from __future__ import annotations

from pathlib import Path

from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from xdart.gui.widgets.controls_panel import (
    ControlsPanel,
    FormRow,
    SectionCard,
)
from xrd_tools.session.readiness import (
    ControlFieldKind,
    ControlFormField,
    SectionId,
)


def _dispose(widget: QtWidgets.QWidget) -> None:
    app = QtWidgets.QApplication.instance()
    widget.close()
    widget.deleteLater()
    if app is not None:
        app.processEvents()


def test_embedded_source_visibility_is_explicit_through_recovery() -> None:
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = ControlsPanel()
    card = SectionCard("direct")
    source = QtWidgets.QLabel("source")
    replacement = QtWidgets.QLabel("replacement")
    direct = QtWidgets.QLabel("direct")
    try:
        assert card.embedded_visible() is False
        card.set_embedded_widget(direct, visible=True)
        assert card.embedded_visible() is True
        card.set_embedded_widget(direct, visible=False)
        assert card.embedded_visible() is False
        card.set_embedded_widget(None)
        assert card.embedded_visible() is False

        panel.set_source_widget(source, visible=False)
        assert panel.source_widget_visible() is False
        assert panel.source_card.embedded_visible() is False

        panel.set_source_widget(source, visible=True)
        assert panel.source_widget_visible() is True
        assert panel.source_card.embedded_visible() is True
        panel.set_source_widget(source, visible=False)
        assert panel.source_widget_visible() is False
        assert panel.source_card.embedded_visible() is False

        panel.set_source_widget(replacement, visible=True)
        assert panel.source_widget_visible() is True
        assert panel.source_card.embedded_visible() is True
        panel.set_source_widget(None)
        assert panel.source_widget_visible() is False
        assert panel.source_card.embedded_visible() is False
    finally:
        _dispose(card)
        _dispose(panel)


def test_combo_reconciliation_replaces_construction_owned_choices() -> None:
    """Accepted PM1 contract: the projected ``ControlFormField.choices`` are
    the authoritative combo vocabulary — construction-time items are removed,
    not preserved, and no per-item metadata migrates across the rebuild."""
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    row = FormRow(
        label="Motor",
        path=("GI", "th_motor"),
        value="Manual",
        kind="combo",
        choices=("Manual", "owner_th", "owner_eta"),
    )
    token = object()
    row.editor.setItemData(1, token, QtCore.Qt.ItemDataRole.UserRole)
    edits: list[tuple[object, object]] = []
    row.valueChanged.connect(
        lambda path, value: edits.append((path, value))
    )
    try:
        field = ControlFormField(
            section=SectionId.EXPERIMENT,
            label="Motor",
            path=("GI", "th_motor"),
            value="projection_only",
            kind=ControlFieldKind.COMBO,
            choices=("Manual", "projection_only"),
        )

        assert row.apply_field(field) is True
        visible = tuple(
            row.editor.itemText(index)
            for index in range(row.editor.count())
        )
        assert "owner_th" not in visible
        assert "owner_eta" not in visible
        assert visible == ("Manual", "projection_only")
        assert row.editor.itemData(
            0, QtCore.Qt.ItemDataRole.UserRole
        ) is not token
        assert row.editor.itemData(
            1, QtCore.Qt.ItemDataRole.UserRole
        ) is not token
        assert row.current_value() == "projection_only"
        assert edits == []
    finally:
        _dispose(row)


def test_combo_commits_only_user_activation_after_popup_event_turn() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    row = FormRow(
        label="Source",
        path=("Signal", "inp_type"),
        value="Image Series",
        kind="combo",
        choices=("Image Series", "Image Directory"),
    )
    edits: list[tuple[object, object]] = []
    row.valueChanged.connect(
        lambda path, value: edits.append((path, value))
    )
    try:
        row.editor.setCurrentText("Image Directory")
        assert edits == []

        row.editor.textActivated.emit("Image Directory")
        assert edits == []
        app.processEvents()

        assert edits == [
            (("Signal", "inp_type"), "Image Directory")
        ]
    finally:
        _dispose(row)


def test_filename_presentation_retains_exact_model_then_commits_typed_edit() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    host = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(host)
    full_path = "/beamline/raw/run.with.dots/image_0001.tif"
    row = FormRow(
        label="Image File",
        path=("Signal", "File"),
        value=full_path,
        browse=True,
    )
    focus_sink = QtWidgets.QLineEdit()
    layout.addWidget(row)
    layout.addWidget(focus_sink)
    edits: list[tuple[object, object]] = []
    row.valueChanged.connect(
        lambda path, value: edits.append((path, value))
    )
    try:
        assert row.editor.text() == Path(full_path).name
        assert row.editor.toolTip() == full_path
        assert row.toolTip() == full_path
        assert row.current_value() == full_path

        manual = "../raw/relocated_0002.tif"
        host.show()
        app.processEvents()
        row.editor.setFocus()
        app.processEvents()
        assert row.editor.hasFocus()
        row.editor.selectAll()
        QtTest.QTest.keyClicks(row.editor, manual)
        app.processEvents()
        focus_sink.setFocus()
        app.processEvents()
        assert edits == [(("Signal", "File"), manual)]
        assert row.current_value() == manual
    finally:
        _dispose(host)
