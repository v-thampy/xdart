"""Offscreen tests for the native Controls panel."""

import gc
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtCore, QtWidgets
from pyqtgraph.Qt import QtTest


def _user_types(qapp, widget, editor, text):
    """Drive a REAL user edit into *editor*.

    §19.9: a programmatic ``editor.setText(...)`` on an unfocused row is a
    PROJECTION, not user intent — the no-focus full-form sweep that used to
    convert it into a journal entry is deleted.  The production input
    authorities are the keystroke draft signal (``textEdited`` →
    ``draftChanged``) and the focused-editor flush, and real focus + real
    ``keyClicks`` drives both, exactly as the amended §12 harvest tests do.
    """
    widget.show()
    qapp.processEvents()
    editor.setFocus()
    qapp.processEvents()
    assert editor.hasFocus()
    editor.selectAll()
    QtTest.QTest.keyClicks(editor, text)
    qapp.processEvents()

from xrd_tools.session.readiness import (
    ControlAction,
    ControlFieldKind,
    ControlFormField,
    ControlsProjection,
    ProcessingPage,
    SectionId,
    build_native_int_reduction_plan_from_args,
    build_native_int_reduction_plan_from_scan,
)
from xdart.gui.widgets.controls_panel import (
    ControlsPanel,
    FormRow,
    PillRow,
    RangeRow,
    SegmentedControl,
    SubsectionCard,
)


@pytest.fixture(autouse=True)
def _controls_panel_session_isolation():
    """Keep saved integrator state from leaking between tests in this module.

    Keep any saved control session from leaking between standalone panel tests.
    """

    path = os.environ.get("XDART_SESSION_FILE")

    def _unlink_session():
        if not path:
            return
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass

    _unlink_session()
    yield
    _unlink_session()


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _drain_qt_events_after_test(qapp):
    yield
    for _ in range(3):
        qapp.processEvents()
    gc.collect()
    for _ in range(2):
        qapp.processEvents()


def test_controls_panel_renders_bound_render_state_directly(qapp):
    state = ControlsProjection(
        processing_page=ProcessingPage.INT_2D,
        fields=(
            ControlFormField(
                section=SectionId.PROJECT,
                label="Folder",
                path=("Project", "project_folder"),
                value="/data",
                browse=True,
            ),
            ControlFormField(
                section=SectionId.SOURCE,
                label="Source",
                path=("Signal", "inp_type"),
                value="Image Series",
                kind=ControlFieldKind.COMBO,
                choices=("Image Series", "Image Directory"),
                enabled=False,
                reason="locked",
            ),
        ),
        section_actions={},
        detector_summary="",
    )

    panel = ControlsPanel()
    panel.reconcile(state)

    project_rows = panel.project_card.body.findChildren(FormRow)
    source_rows = panel.source_card.body.findChildren(FormRow)
    assert [row.label.text() for row in project_rows] == ["Folder"]
    assert [row.label.text() for row in source_rows] == ["Source"]
    assert not source_rows[0].editor.isEnabled()
    assert source_rows[0].toolTip() == "locked"


def test_controls_panel_detector_status_uses_poni_summary(qapp):
    state = ControlsProjection(
        processing_page=ProcessingPage.INT_2D,
        fields=(
            ControlFormField(
                section=SectionId.EXPERIMENT,
                label="Poni",
                path=("Signal", "poni_file"),
                value="/tmp/example.poni",
                browse=True,
            ),
        ),
        section_actions={},
        detector_summary="Eiger 1M · 200.4mm · fitted",
    )

    panel = ControlsPanel()
    panel.reconcile(state)

    detector = next(
        card for card in panel.experiment_card.body.findChildren(SubsectionCard)
        if card.title.text() == "Detector"
    )
    assert detector.status.text() == "Eiger 1M · 200.4mm · fitted"


def test_controls_panel_viewer_mode_shows_only_project(qapp):
    state = ControlsProjection(
        processing_page=ProcessingPage.VIEWER,
        fields=(
            ControlFormField(
                section=SectionId.PROJECT,
                label="Folder",
                path=("Project", "project_folder"),
                value="/data",
                browse=True,
            ),
            ControlFormField(
                section=SectionId.SOURCE,
                label="Source",
                path=("Signal", "inp_type"),
                value="Image Series",
                kind=ControlFieldKind.COMBO,
            ),
            ControlFormField(
                section=SectionId.EXPERIMENT,
                label="Poni",
                path=("Signal", "poni_file"),
                value="",
                browse=True,
            ),
            ControlFormField(
                section=SectionId.PROCESSING,
                label="Background",
                path=("BG", "bg_type"),
                value="None",
                kind=ControlFieldKind.COMBO,
            ),
        ),
        section_actions={},
        detector_summary="",
    )

    panel = ControlsPanel()
    panel.reconcile(state)

    assert not panel.project_card.isHidden()
    assert panel.source_card.isHidden()
    assert panel.experiment_card.isHidden()
    assert panel.processing_card.isHidden()


def test_controls_panel_hides_setup_cards_until_project_is_selected(qapp):
    def projection(project_root):
        return ControlsProjection(
            processing_page=ProcessingPage.INT_2D,
            fields=(
                ControlFormField(
                    section=SectionId.PROJECT,
                    label="Folder",
                    path=("Project", "project_folder"),
                    value=project_root,
                    browse=True,
                ),
                ControlFormField(
                    section=SectionId.SOURCE,
                    label="Source",
                    path=("Signal", "inp_type"),
                    value="Image Series",
                    kind=ControlFieldKind.COMBO,
                ),
                ControlFormField(
                    section=SectionId.EXPERIMENT,
                    label="Poni",
                    path=("Signal", "poni_file"),
                    value="",
                    browse=True,
                ),
                ControlFormField(
                    section=SectionId.PROCESSING,
                    label="Background",
                    path=("BG", "bg_type"),
                    value="None",
                    kind=ControlFieldKind.COMBO,
                ),
            ),
            section_actions={},
            detector_summary="",
        )

    panel = ControlsPanel()
    panel.reconcile(projection(""))

    assert not panel.project_card.isHidden()
    assert panel.source_card.isHidden()
    assert panel.experiment_card.isHidden()
    assert panel.processing_card.isHidden()

    assert panel.reconcile(projection("/data"))
    assert not panel.source_card.isHidden()
    assert not panel.experiment_card.isHidden()
    assert not panel.processing_card.isHidden()

    assert panel.reconcile(projection(""))
    assert panel.source_card.isHidden()
    assert panel.experiment_card.isHidden()
    assert panel.processing_card.isHidden()


def test_run_readiness_label_elides_without_widening_controls(qapp):
    from xdart.gui.widgets.run_controls import RunControlsBar

    controls = RunControlsBar()
    try:
        controls.set_readiness_summary(
            "Needs setup · short blocker",
            ready=False,
            tooltip="short blocker",
        )
        controls.resize(360, controls.sizeHint().height())
        controls.show()
        qapp.processEvents()
        short_hint = controls.sizeHint().width()

        long_reason = "Needs setup · " + ("blocked configuration detail " * 20)
        controls.set_readiness_summary(
            long_reason,
            ready=False,
            tooltip=long_reason,
        )
        controls.resize(360, controls.sizeHint().height())
        qapp.processEvents()

        assert controls.sizeHint().width() == short_hint
        assert controls.readinessLabel.text() != long_reason
        assert controls.readinessLabel.text().endswith("…")
        assert controls.readinessLabel.toolTip() == long_reason
    finally:
        controls.close()
        controls.deleteLater()


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("Signal", "File"), "/data/raw/scan_0001.nxs"),
        (("Signal", "img_dir"), "/data/raw/images"),
        (("Signal", "poni_file"), "/data/calibration/cal.poni"),
        (("Signal", "mask_file"), "/data/calibration/mask.edf"),
        (("Project", "h5_dir"), "/data/processed/results"),
    ),
)
def test_compact_path_expands_for_direct_edit_then_collapses(
    qapp, path, value,
):
    row = FormRow(
        label="Path",
        path=path,
        value=value,
        browse=True,
    )
    row.show()
    qapp.processEvents()
    row.editor.clearFocus()
    qapp.processEvents()
    qapp.processEvents()
    edits = []
    drafts = []
    row.valueChanged.connect(lambda changed, text: edits.append((changed, text)))
    row.draftChanged.connect(lambda changed, text: drafts.append((changed, text)))
    try:
        assert row.editor.text() == Path(value).name
        assert row.current_value() == value

        row.editor.setFocus()
        qapp.processEvents()
        assert row.editor.text() == value
        assert row.current_value() == value
        assert edits == []
        assert drafts == []

        replacement = "/edited/directly/new-source.nxs"
        row.editor.selectAll()
        QtTest.QTest.keyClicks(row.editor, replacement)
        qapp.processEvents()
        assert row.current_value() == replacement
        assert drafts and drafts[-1] == (path, replacement)

        row.editor.clearFocus()
        qapp.processEvents()
        qapp.processEvents()
        assert edits[-1] == (path, replacement)
        assert row.current_value() == replacement
        assert row.editor.text() == Path(replacement).name
        assert row.editor.toolTip() == replacement
    finally:
        row.close()
        row.deleteLater()


def test_project_folder_remains_full_path_while_focused(qapp):
    value = "/data/very/long/project"
    row = FormRow(
        label="Folder",
        path=("Project", "project_folder"),
        value=value,
        browse=True,
    )
    row.show()
    qapp.processEvents()
    try:
        assert row.editor.text() == value
        row.editor.setFocus()
        qapp.processEvents()
        assert row.editor.text() == value
        row.editor.clearFocus()
        qapp.processEvents()
        assert row.editor.text() == value
    finally:
        row.close()
        row.deleteLater()


def test_rejected_compact_path_edit_restores_authoritative_path(qapp):
    path = ("Signal", "File")
    original = "/data/raw/scan_0001.nxs"
    row = FormRow(
        label="Image File",
        path=path,
        value=original,
        browse=True,
    )
    authoritative = ControlFormField(
        SectionId.SOURCE,
        "Image File",
        path,
        original,
        browse=True,
    )
    row.valueChanged.connect(
        lambda _path, _value: row.apply_field(authoritative)
    )
    row.show()
    qapp.processEvents()
    try:
        row.editor.setFocus()
        qapp.processEvents()
        row.editor.selectAll()
        QtTest.QTest.keyClicks(row.editor, "/rejected/missing.nxs")
        QtTest.QTest.keyClick(
            row.editor, QtCore.Qt.Key.Key_Return
        )
        qapp.processEvents()
        assert row.current_value() == original

        row.editor.clearFocus()
        qapp.processEvents()
        qapp.processEvents()
        assert row.current_value() == original
        assert row.editor.text() == Path(original).name
        assert row.editor.toolTip() == original
    finally:
        row.close()
        row.deleteLater()


@pytest.mark.parametrize("theme", ("dark", "light"))
def test_controls_panel_checked_disabled_matches_disabled_text_color(
    theme,
):
    from xdart.gui.themes import render_qss

    qss = render_qss(theme)

    def color_for(selector):
        start = qss.index(selector)
        body = qss[start:qss.index("}", start)]
        return next(
            line.strip() for line in body.splitlines()
            if line.strip().startswith("color:")
        )

    assert color_for(
        "QPushButton#controlsToggleButton:checked:disabled"
    ) == color_for("QPushButton#controlsToggleButton:disabled")



def test_reconcile_rebuilds_when_fields_appear(qapp):
    """LV-UI-5: Standard→Grazing ADDS the θ-motor field; the in-place fast
    path must refuse (keys changed) so the full render mounts the new row —
    public reconcile rebuilds once when the schema changes."""
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_inventory import GI_MOTOR
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase

    panel = ControlsPanel()
    try:
        intent = RunIntent()
        panel.reconcile(project_controls(
            RunIntentStore(intent).snapshot(), None, RunPhase.IDLE))
        assert not [r for r in panel.findChildren(FormRow)
                    if tuple(r.path) == GI_MOTOR]
        panel.experiment_card.set_status_text("standard")
        panel.experiment_card.set_valid_marker(True, "ready")

        intent.gi.enabled = True
        grazing = project_controls(
            RunIntentStore(intent).snapshot(), None, RunPhase.IDLE)
        assert panel.reconcile(grazing) is False
        rows = [r for r in panel.findChildren(FormRow)
                if tuple(r.path) == GI_MOTOR]
        assert rows, "theta-motor row must mount on the Grazing switch"
        assert panel.experiment_card.status.text() == "standard"
        assert not panel.experiment_card.valid_marker.isHidden()
        assert panel.experiment_card.valid_marker.toolTip() == "ready"
    finally:
        panel.close()
        panel.deleteLater()


def test_reconcile_updates_action_buttons_without_spurious_repolish(
    qapp, monkeypatch,
):
    """Unchanged/action-only refreshes avoid polish; style changes do not."""
    from dataclasses import replace

    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase
    from xdart.gui.widgets.controls_panel import ActionButton

    store = RunIntentStore(RunIntent())
    disabled = project_controls(
        store.snapshot(), None, RunPhase.IDLE,
        reintegrate_available=False,
    )
    enabled = project_controls(
        store.snapshot(), None, RunPhase.IDLE,
        reintegrate_available=True,
    )
    panel = ControlsPanel()
    try:
        panel.reconcile(disabled)
        before = next(
            button
            for button in panel.findChildren(ActionButton)
            if button.spec.action is ControlAction.REINTEGRATE_1D
        )
        row_before = panel.findChildren(FormRow)[0]
        editor_before = row_before.editor
        assert not before.isEnabled()
        assert "stable processed Browse artifact" in before.toolTip()
        calibrate_before = next(
            button
            for button in panel.findChildren(ActionButton)
            if button.spec.action is ControlAction.CALIBRATE
        )
        assert calibrate_before.text() == "⌖ Calibrate"

        assert panel.reconcile(enabled) is True
        qapp.processEvents()
        after = next(
            button
            for button in panel.findChildren(ActionButton)
            if button.spec.action is ControlAction.REINTEGRATE_1D
            and button.isEnabled()
        )
        assert after is before
        assert panel.findChildren(FormRow)[0] is row_before
        assert row_before.editor is editor_before
        assert "Creates a new immutable 1-D version" in after.toolTip()
        assert "selected artifact remains unchanged" in after.toolTip()

        style = after.style()
        original_unpolish = style.unpolish
        original_polish = style.polish
        repolished: list[tuple[str, ActionButton]] = []

        def record_unpolish(widget):
            if isinstance(widget, ActionButton):
                repolished.append(("unpolish", widget))
            return original_unpolish(widget)

        def record_polish(widget):
            if isinstance(widget, ActionButton):
                repolished.append(("polish", widget))
            return original_polish(widget)

        monkeypatch.setattr(style, "unpolish", record_unpolish)
        monkeypatch.setattr(style, "polish", record_polish)

        # A controls-only refresh commonly projects an equal immutable state.
        # It must neither rebuild nor force every action through the style
        # engine: that global repolish is visible as a whole-panel flicker.
        assert panel.reconcile(enabled) is True
        assert repolished == []
        assert next(
            button
            for button in panel.findChildren(ActionButton)
            if button.spec.action is ControlAction.REINTEGRATE_1D
        ) is after

        active = project_controls(
            store.snapshot(), None, RunPhase.IDLE,
            operation_busy=True,
            calibration_active=True,
            reintegrate_available=True,
        )
        assert panel.reconcile(active) is True
        assert next(
            button
            for button in panel.findChildren(ActionButton)
            if button.spec.action is ControlAction.CALIBRATE
        ) is calibrate_before
        assert calibrate_before.text() == "Cancel Calibration"
        assert repolished == []

        # ``productionReady`` participates in the stylesheet selector.  A
        # genuine change to it must still update the dynamic property and
        # repolish exactly that one action button.
        actions = dict(active.section_actions)
        actions[SectionId.EXPERIMENT] = tuple(
            replace(spec, production_ready=False)
            if spec.action is ControlAction.CALIBRATE
            else spec
            for spec in actions[SectionId.EXPERIMENT]
        )
        style_changed = replace(
            active,
            section_actions=actions,
        )
        assert panel.reconcile(style_changed) is True
        assert calibrate_before.property("productionReady") is False
        assert repolished == [
            ("unpolish", calibrate_before),
            ("polish", calibrate_before),
        ]

        actions = dict(style_changed.section_actions)
        processing = actions[SectionId.PROCESSING]
        actions[SectionId.PROCESSING] = tuple(reversed(processing))
        reordered = replace(
            style_changed,
            section_actions=actions,
        )
        assert panel.reconcile(reordered) is False
    finally:
        panel.close()
        panel.deleteLater()


def test_reconcile_locks_gi_more_without_rebuilding(qapp):
    """A controls-only operation refresh must not leave the GI popup live."""
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase

    intent = RunIntent()
    intent.gi.enabled = True
    store = RunIntentStore(intent)
    idle = project_controls(store.snapshot(), None, RunPhase.IDLE)
    busy = project_controls(
        store.snapshot(), None, RunPhase.IDLE,
        operation_busy=True,
        calibration_active=True,
    )
    panel = ControlsPanel()
    try:
        panel.reconcile(idle)
        more = next(
            button
            for button in panel.experiment_card.body.findChildren(
                QtWidgets.QToolButton
            )
            if button.objectName() == "controlsMoreButton"
        )
        first_row = panel.findChildren(FormRow)[0]
        more.click()
        qapp.processEvents()
        assert panel._gi_options_popup is not None

        assert panel.reconcile(busy) is True
        qapp.processEvents()
        assert more is next(
            button
            for button in panel.experiment_card.body.findChildren(
                QtWidgets.QToolButton
            )
            if button.objectName() == "controlsMoreButton"
        )
        assert panel.findChildren(FormRow)[0] is first_row
        assert not more.isEnabled()
        assert panel._gi_options_popup is None
        assert {
            subsection.status.text()
            for subsection in panel.processing_card.body.findChildren(
                SubsectionCard
            )
        } == {"locked"}

        more.click()
        qapp.processEvents()
        assert panel._gi_options_popup is None
    finally:
        panel.close()
        panel.deleteLater()


def test_reconcile_refreshes_source_energy_popup_capture(qapp):
    """The Source More button must open the newly projected preference."""
    from dataclasses import replace

    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase

    base = project_controls(
        RunIntentStore(RunIntent()).snapshot(), None, RunPhase.IDLE,
    )
    energy = ControlFormField(
        SectionId.SOURCE,
        "Energy Source",
        ("Source", "energy_preference"),
        "poni",
        kind=ControlFieldKind.COMBO,
        choices=("poni", "metadata"),
    )
    before = replace(
        base,
        fields=base.fields + (energy,),
    )
    after = replace(
        before,
        fields=tuple(
            replace(field, value="metadata")
            if field.path == energy.path
            else field
            for field in before.fields
        ),
    )
    panel = ControlsPanel()
    try:
        panel.reconcile(before)
        button = next(
            candidate
            for candidate in panel.source_card.body.findChildren(
                QtWidgets.QToolButton
            )
            if candidate.property("role") == "sourceEnergy"
        )
        assert panel.reconcile(after) is True
        assert button is next(
            candidate
            for candidate in panel.source_card.body.findChildren(
                QtWidgets.QToolButton
            )
            if candidate.property("role") == "sourceEnergy"
        )

        button.click()
        qapp.processEvents()
        popup = panel._source_energy_popup
        assert popup is not None
        assert type(popup) is QtWidgets.QWidget
        assert popup.objectName() == "controlsEnergyPopup"
        segmented = popup.findChild(SegmentedControl)
        assert segmented is not None
        assert segmented.current_value() == "metadata"
    finally:
        panel.close()
        panel.deleteLater()


def test_controls_theme_selectors_match_current_runtime_widget_classes():
    from xdart.gui.themes import render_qss

    qss = render_qss("dark")
    assert (
        "QWidget#controlsEnergyPopup,\n"
        "QWidget#controlsGIMorePopup {"
    ) in qss
    assert "QWidget#controlsActionRow {" in qss
    assert "QMenu#controlsEnergyPopup" not in qss
    assert "QFrame#controlsActionRow" not in qss


def test_reconcile_refreshes_derived_subsection_statuses(qapp):
    from dataclasses import replace

    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase

    before = project_controls(
        RunIntentStore(RunIntent()).snapshot(), None, RunPhase.IDLE,
    )
    after = replace(
        before,
        detector_summary="Eiger4M · fitted",
    )
    panel = ControlsPanel()
    try:
        panel.reconcile(before)
        detector = next(
            subsection
            for subsection in panel.experiment_card.body.findChildren(
                SubsectionCard
            )
            if subsection.title.text() == "Detector"
        )
        assert detector.status.text() != "Eiger4M · fitted"
        assert panel.reconcile(after) is True
        assert detector.status.text() == "Eiger4M · fitted"
        assert detector.status.isVisibleTo(detector)
    finally:
        panel.close()
        panel.deleteLater()


def test_reconcile_refuses_fast_path_before_schema_rebuild(qapp):
    from dataclasses import replace

    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase

    before = project_controls(
        RunIntentStore(RunIntent()).snapshot(), None, RunPhase.IDLE,
    )
    target = before.fields[0]
    after = replace(
        before,
        fields=(replace(target, label=target.label + " changed"),)
        + before.fields[1:],
    )
    panel = ControlsPanel()
    try:
        panel.reconcile(before)
        row = next(
            candidate
            for candidate in panel.findChildren(FormRow)
            if candidate.path == target.path
        )
        label = row.label.text()
        assert panel.reconcile(after) is False
        assert panel.projection is after
        assert row.label.text() == label
        replacement = next(
            candidate
            for candidate in panel.findChildren(FormRow)
            if candidate.path == target.path
        )
        assert replacement is not row
        assert replacement.label.text() == target.label + " changed"
    finally:
        panel.close()
        panel.deleteLater()


def test_threshold_and_mask_saturated_are_independent_in_vnext(qapp):
    """Both switches are editable and each emits only its own exact path."""
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_inventory import (
        MASK_SATURATION,
        THRESHOLD_ENABLED,
    )
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase
    from xdart.gui.widgets.controls_panel import PillRow, RangeRow

    panel = ControlsPanel()
    try:
        panel.reconcile(project_controls(
            RunIntentStore(RunIntent()).snapshot(), None, RunPhase.IDLE))
        row = next(
            r for r in panel.findChildren(RangeRow)
            if tuple(r._low_path) == ("Mask", "min")
        )
        tpath, btn = row._toggle
        assert tuple(tpath) == THRESHOLD_ENABLED
        assert not btn.isChecked()                 # manual band is off
        assert not row._low.isEnabled()
        assert not row._high.isEnabled()

        emitted = []
        panel.fieldValueChanged.connect(
            lambda p, v: emitted.append((tuple(p), v)))
        btn.setChecked(True)
        assert (THRESHOLD_ENABLED, True) in emitted

        pill_rows = [
            p for p in panel.findChildren(PillRow)
            if any(tuple(path) == MASK_SATURATION for path, _ in p._pills)
        ]
        assert pill_rows, "the Mask Saturated pill must stay visible"
        pill = next(
            b for path, b in pill_rows[0]._pills
            if tuple(path) == MASK_SATURATION
        )
        assert pill.isEnabled()
        assert pill.isChecked()
        assert pill.objectName() == "controlsPillButton"
        assert (MASK_SATURATION, True) in pill_rows[0].current_edits()
        pill.setChecked(False)
        assert (MASK_SATURATION, False) in emitted
        pill_top = pill_rows[0].layout().contentsMargins().top()
        conditioning = next(
            card
            for card in panel.processing_card.body.findChildren(SubsectionCard)
            if card.title.text() == "Conditioning"
        )
        assert pill_top == 3
        assert (
            conditioning.body_layout.spacing() + pill_top
            == conditioning.body_layout.contentsMargins().bottom()
        )

        # Scope guard (Codex P2): the seeded max is a detector-FAMILY display
        # default, not an acquisition-dtype fact — the RENDERED max-bound
        # widget must say so whether the band is disabled or enabled.
        assert "display default" in row._high.toolTip()

        manual = RunIntent()
        manual.threshold.mask_saturation = False
        manual.threshold.apply_threshold = True
        panel.reconcile(project_controls(
            RunIntentStore(manual).snapshot(), None, RunPhase.IDLE))
        manual_row = next(
            r for r in panel.findChildren(RangeRow)
            if tuple(r._low_path) == ("Mask", "min")
        )
        assert manual_row._toggle[1].isChecked()
        assert manual_row._high.isEnabled()
        assert "display default" in manual_row._high.toolTip()
        manual_pill = next(
            b
            for pills in panel.findChildren(PillRow)
            for path, b in pills._pills
            if tuple(path) == MASK_SATURATION
        )
        assert manual_pill.isEnabled()
        assert not manual_pill.isChecked()
    finally:
        panel.close()
        panel.deleteLater()


def test_threshold_bounds_render_without_decimals_without_rounding_the_model(
    qapp,
):
    """Threshold formatting is presentation-only; edits retain float truth."""

    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase

    def _threshold_row(host):
        return next(
            row for row in host.findChildren(RangeRow)
            if tuple(row._low_path) == ("Mask", "min")
        )

    intent = RunIntent()
    intent.threshold.apply_threshold = True
    intent.threshold.mask_saturation = False
    intent.threshold.threshold_min = 12.0
    intent.threshold.threshold_max = 4294967295.0
    panel = ControlsPanel()
    generic = RangeRow(
        label="Q",
        low={"path": ("Int1D", "radial_low"), "value": 0.25},
        high={"path": ("Int1D", "radial_high"), "value": 4.75},
    )
    try:
        panel.reconcile(project_controls(
            RunIntentStore(intent).snapshot(), None, RunPhase.IDLE
        ))
        row = _threshold_row(panel)
        assert row._low.text() == "12"
        assert row._high.text() == "4294967295"
        assert (("Mask", "min"), "12.0") in row.current_edits()
        assert (
            (("Mask", "max"), "4294967295.0") in row.current_edits()
        )

        emitted = []
        row.valueChanged.connect(
            lambda path, value: emitted.append((tuple(path), value))
        )
        row._low.editingFinished.emit()
        assert emitted[-1] == (("Mask", "min"), "12.0")

        _user_types(qapp, panel, row._low, "12")
        focused = panel.focused_form_edit()
        assert focused is not None
        assert focused.path == ("Mask", "min")
        assert focused.value == "12"
        row._low.editingFinished.emit()
        assert emitted[-1] == (("Mask", "min"), "12")

        row._low.clearFocus()
        qapp.processEvents()
        # A later accepted exact value can share the same rounded display
        # bucket.  It must clear the old draft marker and remain authoritative
        # when the untouched editor is focused/harvested again.
        assert panel.reconcile(project_controls(
            RunIntentStore(intent).snapshot(), None, RunPhase.IDLE
        ))
        row = _threshold_row(panel)
        assert row._low.text() == "12"
        assert (("Mask", "min"), "12.0") in row.current_edits()
        row._low.setFocus()
        qapp.processEvents()
        focused = panel.focused_form_edit()
        assert focused is not None and focused.value == "12.0"
        row._low.clearFocus()
        qapp.processEvents()

        # A deliberately non-integral draft remains truthful after its
        # accepted commit and blur; zero-decimal formatting must never render
        # 1.5 as a different executed value such as 2.
        _user_types(qapp, panel, row._low, "1.5")
        row._low.editingFinished.emit()
        assert emitted[-1] == (("Mask", "min"), "1.5")
        row._low.clearFocus()
        qapp.processEvents()

        updated = RunIntent()
        updated.threshold.apply_threshold = True
        updated.threshold.mask_saturation = False
        updated.threshold.threshold_min = 1.5
        updated.threshold.threshold_max = 100.75
        assert panel.reconcile(project_controls(
            RunIntentStore(updated).snapshot(), None, RunPhase.IDLE
        ))
        row = _threshold_row(panel)
        assert row._low.text() == "1.5"
        assert row._high.text() == "100.75"
        assert (("Mask", "min"), "1.5") in row.current_edits()
        assert (("Mask", "max"), "100.75") in row.current_edits()

        # Other numeric ranges keep their existing precision.
        assert generic._low.text() == "0.25"
        assert generic._high.text() == "4.75"
    finally:
        generic.close()
        generic.deleteLater()
        panel.close()
        panel.deleteLater()


def test_max_bound_scope_caveat_survives_run_lock(qapp):
    """DESIGN_STOP secondary (2026-08-04): the lock reason outranks the
    tooltip table too, so the detector-scope caveat must ride the LOCKED
    max-bound reason — at construction under lock and across in-place
    lock/unlock state updates."""
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase
    from xdart.gui.widgets.controls_panel import RangeRow

    def _row(host):
        return next(
            r for r in host.findChildren(RangeRow)
            if tuple(r._low_path) == ("Mask", "min")
        )

    store = RunIntentStore(RunIntent())
    panel = ControlsPanel()
    try:
        # Construction while run-locked.
        panel.reconcile(project_controls(
            store.snapshot(), None, RunPhase.RUNNING))
        locked = _row(panel)
        assert not locked._high.isEnabled()
        assert "locked" in locked._high.toolTip()
        assert "display default" in locked._high.toolTip()

        # In-place unlock, then re-lock, through the update path.
        panel.reconcile(project_controls(
            store.snapshot(), None, RunPhase.IDLE))
        assert "display default" in _row(panel)._high.toolTip()
        panel.reconcile(project_controls(
            store.snapshot(), None, RunPhase.RUNNING))
        relocked = _row(panel)
        assert "locked" in relocked._high.toolTip()
        assert "display default" in relocked._high.toolTip()
    finally:
        panel.close()
        panel.deleteLater()


def test_range_toggle_presents_manual_on_while_preserving_auto_model(qapp):
    """A lit range toggle means its boxes apply; storage remains ``*_auto``."""
    from xdart.gui.widgets.controls_panel import RangeRow

    emitted = []
    row = RangeRow(
        label="Q",
        low={"path": ("Int1D", "radial_low"), "value": 0.0},
        high={"path": ("Int1D", "radial_high"), "value": 5.0},
        toggle={"path": ("Int1D", "radial_auto"), "value": True},
    )
    row.valueChanged.connect(lambda p, v: emitted.append((tuple(p), v)))
    try:
        btn = row._toggle[1]
        assert not btn.isChecked()                     # model is Auto
        assert (("Int1D", "radial_auto"), True) in row.current_edits()
        btn.setChecked(True)                           # entered bounds ON
        assert emitted == [(("Int1D", "radial_auto"), False)]
        assert (("Int1D", "radial_auto"), False) in row.current_edits()
    finally:
        row.close()
        row.deleteLater()


@pytest.mark.parametrize(
    "path",
    (
        ("Int1D", "radial_auto"),
        ("Int1D", "azim_auto"),
        ("Int2D", "radial_auto"),
        ("Int2D", "azim_auto"),
    ),
)
def test_all_integration_range_toggles_invert_only_the_ui_fact(qapp, path):
    """Every integration range uses lit=manual without changing storage."""

    stem = path[-1].removesuffix("_auto")
    low_path = (path[0], f"{stem}_low")
    high_path = (path[0], f"{stem}_high")
    row = RangeRow(
        label=stem,
        low={"path": low_path, "value": 0.0},
        high={"path": high_path, "value": 5.0},
        toggle={"path": path, "value": True},
    )
    try:
        button = row._toggle[1]
        assert not button.isChecked()
        assert "entered range" in button.toolTip()
        fields = {
            low_path: ControlFormField(
                SectionId.PROCESSING, "Low", low_path, 1.0,
            ),
            high_path: ControlFormField(
                SectionId.PROCESSING, "High", high_path, 4.0,
            ),
            path: ControlFormField(
                SectionId.PROCESSING, "Manual", path, False,
                kind=ControlFieldKind.BOOL,
            ),
        }
        assert row.apply_fields(fields)
        assert button.isChecked()
        assert "entered range" in button.toolTip()
        assert (path, False) in row.current_edits()

        fields[path] = ControlFormField(
            SectionId.PROCESSING, "Manual", path, True,
            kind=ControlFieldKind.BOOL,
        )
        assert row.apply_fields(fields)
        assert not button.isChecked()
        assert (path, True) in row.current_edits()
    finally:
        row.close()
        row.deleteLater()


def test_threshold_toggle_keeps_direct_apply_polarity(qapp):
    """Threshold's direct enable stays checked exactly when the band applies."""
    from xdart.gui.widgets.controls_panel import RangeRow

    emitted = []
    row = RangeRow(
        label="Threshold",
        low={"path": ("Mask", "min"), "value": None},
        high={"path": ("Mask", "max"), "value": None},
        toggle={"path": ("Mask", "Threshold"), "value": False},
    )
    row.valueChanged.connect(lambda p, v: emitted.append((tuple(p), v)))
    try:
        btn = row._toggle[1]
        assert not btn.isChecked()                    # off -> not applied
        assert (("Mask", "Threshold"), False) in row.current_edits()
        btn.setChecked(True)                          # user enables threshold
        assert emitted == [(("Mask", "Threshold"), True)]
        assert (("Mask", "Threshold"), True) in row.current_edits()
    finally:
        row.close()
        row.deleteLater()

# ── SW-7 (§10): wrangler swap resyncs the θ-motor choices to the owner ─────


# ── SW-8 (§10): Axis edits keep the gi_config mode copy in sync ─────────────


# ── POL-1: fresh-scan polarization default ON at 0.99 ──────────────────────


# ── Live chip + waiting status in the readiness bar (2026-07-13) ────────────


# ── DIR-1: the outgoing scan's frame list paints before the boundary clear ──


# ---------------------------------------------------------------------------
# T-2R (§14.11) — checked-commit atomicity: preflight, push-before-setter,
# rollback-all, contained exceptions, display-scan rollback, typed attempt
# result.  Each is RED at 1f8b4255 and GREEN at the T-2R correction.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# T-2.1 (§13.6/§13.7/§13.8/§13.11 + §14.11.B.4/D.2) — compound PONI carrier,
# source-token GI hydration, child-only-dir UNKNOWN, candidate SourceSpec.
# Each is RED at 1f8b4255 and GREEN at the T-2.1 correction.
# ---------------------------------------------------------------------------


def _write_poni(path, dist):
    path.write_text(
        f"Distance: {dist}\nPoni1: 0.01\nPoni2: 0.02\n"
        "Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1.0e-10\n"
    )
    return str(path)


def test_section_number_presentation_option_preserves_canonical_default(qapp):
    compact = ControlsPanel(show_section_numbers=False)
    canonical = ControlsPanel()
    try:
        canonical_chips = canonical.findChildren(
            QtWidgets.QLabel, "controlsSectionChip")
        compact_chips = compact.findChildren(
            QtWidgets.QLabel, "controlsSectionChip")

        assert any(
            chip.text() == "1" and not chip.isHidden()
            for chip in canonical_chips)
        assert all(chip.isHidden() for chip in compact_chips)
    finally:
        canonical.close()
        canonical.deleteLater()
        compact.close()
        compact.deleteLater()


_COMBO_PATH = ("gi", "motor")


def _combo_row(*, value="Manual", choices=("Manual",)):
    return FormRow(
        label="Motor", path=_COMBO_PATH, value=value,
        kind="combo", choices=choices,
    )


def _combo_field(choices, value="Manual"):
    return ControlFormField(
        SectionId.EXPERIMENT, "Motor", _COMBO_PATH, value,
        ControlFieldKind.COMBO, tuple(choices),
    )


def _combo_items(row):
    return tuple(
        row.editor.itemText(index) for index in range(row.editor.count())
    )


def test_form_row_combo_reconciles_added_removed_and_reordered_choices(qapp):
    """Projected ``field.choices`` are the authoritative combo vocabulary."""
    row = _combo_row()
    emitted = []
    row.valueChanged.connect(lambda *values: emitted.append(values))
    try:
        assert row.apply_field(_combo_field(("Manual", "th", "eta")))
        assert _combo_items(row) == ("Manual", "th", "eta")

        assert row.apply_field(_combo_field(("Manual", "eta")))
        assert _combo_items(row) == ("Manual", "eta")

        assert row.apply_field(_combo_field(("eta", "Manual")))
        assert _combo_items(row) == ("eta", "Manual")

        assert row.editor.currentText() == "Manual"
        qapp.processEvents()
        assert emitted == []
    finally:
        row.close()
        row.deleteLater()


def test_form_row_combo_appends_absent_explicit_current_last(qapp):
    """An absent explicit current value is appended LAST, typed order kept."""
    row = _combo_row()
    try:
        assert row.apply_field(_combo_field(("Manual", "eta"), value="halpha"))
        assert _combo_items(row) == ("Manual", "eta", "halpha")
        assert row.editor.currentText() == "halpha"
    finally:
        row.close()
        row.deleteLater()


def test_form_row_combo_reconciliation_emits_no_field_value_changed(qapp):
    """A rebuild is silent on ``fieldValueChanged``; a real pick still emits."""
    panel = ControlsPanel()
    row = _combo_row(value="halpha", choices=("Manual", "halpha"))
    row.valueChanged.connect(panel.fieldValueChanged)
    emitted = []
    panel.fieldValueChanged.connect(lambda *values: emitted.append(values))
    try:
        assert row.apply_field(_combo_field(("Manual", "eta"), value="eta"))
        qapp.processEvents()
        assert emitted == []
        assert row.editor.currentText() == "eta"

        # E4-accepted seam: ANY programmatic index/text set stays silent (a
        # synchronous rebuild from currentTextChanged could destroy a native
        # popup mid-open); only a genuine user activation commits, deferred
        # one event-loop turn.
        row.editor.setCurrentIndex(0)
        qapp.processEvents()
        assert emitted == []
        row.editor.textActivated.emit("Manual")
        qapp.processEvents()
        assert emitted == [(_COMBO_PATH, "Manual")]
    finally:
        row.close()
        row.deleteLater()
        panel.close()
        panel.deleteLater()


def test_form_row_combo_restores_a_previously_blocked_signal_state(qapp):
    """``apply_field`` restores the PRIOR blocked state, never force-clears."""
    row = _combo_row()
    try:
        row.editor.blockSignals(True)
        assert row.apply_field(_combo_field(("Manual", "eta"), value="eta"))
        assert row.editor.signalsBlocked() is True
    finally:
        row.close()
        row.deleteLater()
