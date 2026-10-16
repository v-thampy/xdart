"""COVERAGE-EXPORT-FOLLOWUP-20261014 Stage C, frozen before the fix: clearer coverage controls, the same behavior.

COVERAGE-CONTROL-CLARITY, with the wordings the maintainer approved on October 14 ("Approve all four (Recommended)"):
- the shared recording pill reads "Record angular coverage: On" or "...: Off", and its accessible name stays the
  label;
- Plot Options shows "Show angular coverage", a "2D pixel coverage" section with "Shade uncovered bins", and
  "Detector coordinate grid".
Keys, defaults and behavior are unchanged, and each tooltip says what its control does and when it is unavailable.
LIVE-TEST-FIXES-20261015 LF-007 relocates recording to Advanced → Integrate 1D in both pages, preserving
Int's Okay/Cancel submission and Stitch's immediate editing.
These rows use real pages, real clicks and the shared owners' own reconciliation. Behavior beyond them stays pinned
by the recording, strip, plot-options, shading and grid files.

- C1: Int's and Stitch's recording pill says On or Off through clicks, a Stitch save and reopen, and the shared
  owner's reconciliation, a busy lock included. The bool is unchanged and recording never changes the display choice;
  the default is Off since INT-PERF-REGRESSION-20261014, so a fresh pill says Off and an explicit On comes first.
- C2: "Show angular coverage" is a display choice, and its tooltip says so. Cancel discards it; Okay applies it, as
  a plot option only.
- C3: "2D pixel coverage" and "Shade uncovered bins" are unavailable for an ordinary Int result. The reason says it
  needs a Stitch intensity map with pixel coverage, unlike the 1-D angular coverage. Once available, the control
  keeps its purpose.
- C4: "Detector coordinate grid" keeps the same choices. Its tooltip names the raw detector frame, Run q and χ and
  saved q-only, and gives the precise reason when it is unavailable.
"""
from dataclasses import replace

from pyqtgraph.Qt import QtCore, QtWidgets

from tests.xdart.scattering.e3_shell_support import make_shell_projection
from tests.xdart.scattering.test_angular_support_strip import _masked_series
from tests.xdart.scattering.test_browse_reference_rings_run import _page
from tests.xdart.scattering.test_e4_plot_options import _reconcile
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401

LABEL = "Record angular coverage"


def _pill(widget):
    """The shared recording control by its accessible name, which stays the label while its text adds the state."""
    found = [button for button in widget.findChildren(QtWidgets.QAbstractButton)
             if button.isCheckable() and button.accessibleName() == LABEL]
    assert len(found) == 1, f"{len(found)} controls named '{LABEL}' for assistive technology"
    return found[0]


def _int_recording(qapp, page, *, toggle=False, accept=True):
    """Use the real modal Advanced action; Cancel discards its local recording edit."""
    from xdart.gui.widgets.controls_panel import ActionButton
    from xrd_tools.session.readiness import ControlAction
    assert not [button for button in page._shell.controls.findChildren(QtWidgets.QAbstractButton)
                if button.accessibleName() == LABEL], 'recording belongs in Advanced, not the main 1-D card'
    advanced = next(button for button in page._shell.controls.findChildren(ActionButton)
                    if button.spec.action is ControlAction.ADVANCED_PROCESSING)
    found, errors = [], []

    def edit():
        dialog = page._advanced_dialog
        try:
            pill = _pill(dialog)
            group = pill.parentWidget()
            while group is not None and not isinstance(group, QtWidgets.QGroupBox):
                group = group.parentWidget()
            assert group is not None and group.title() == 'Integrate 1D'
            found.append(pill)
            if toggle:
                pill.click()
        except Exception as error:
            errors.append(error)
        finally:
            if dialog is not None:
                dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok if accept and not errors
                                      else QtWidgets.QDialogButtonBox.StandardButton.Cancel).click()

    QtCore.QTimer.singleShot(0, edit)
    advanced.click()
    qapp.processEvents()
    assert not errors, str(errors)
    assert len(found) == 1
    return found[0]


def _options(qapp):
    """A shown ScientificView with an ordinary Int projection, its Plot Options opened by the real button."""
    from xdart.gui.tabs.scattering.scientific_view import ScientificView
    view = ScientificView()
    state = make_shell_projection(plot_mode="Overlay")
    commands = []
    view.commandRequested.connect(commands.append)
    _reconcile(view, state.scientific, state.navigation)
    view.options.click()
    qapp.processEvents()
    return view, view.plot_options_dialog, commands


def test_c1_the_recording_pill_says_on_or_off_in_int_and_stitch(qapp, tmp_path):
    from tests.xdart.pages.test_stitch_page import _Status
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.tabs.scattering.controls_inventory import RECORD_ANGULAR_SUPPORT
    from xdart.gui.tabs.stitch.page import StitchPage
    from xdart.gui.tabs.stitch.settings import apply_page_settings, page_settings
    from xdart.gui.widgets.controls_panel import PillRow
    from xrd_tools.analysis.angular_support import DEFAULT_RECORD_ANGULAR_SUPPORT
    from xrd_tools.session.readiness import ControlFieldKind, ControlFormField, SectionId
    # INT-PERF-REGRESSION-20261014 Stage A: Off by default. A fresh pill says Off; an explicit On comes first, then the
    # same On -> Off -> On transitions as before.
    assert DEFAULT_RECORD_ANGULAR_SUPPORT is False
    intent = _masked_series(tmp_path / "int")
    del intent.run_options["record_angular_support"]  # a fresh Int page
    with _page(qapp, intent) as (page, view, _observed):
        pill = _int_recording(qapp, page)
        shown = view.plot_options_dialog.projection.show_angular_support
        assert not pill.isChecked() and pill.text() == f"{LABEL}: Off"
        revision = page._intents.revision
        _int_recording(qapp, page, toggle=True, accept=False)
        assert page._intents.revision == revision, 'Cancel discards the local toggle'
        pill = _int_recording(qapp, page, toggle=True)
        assert pill.isChecked() and pill.text() == f"{LABEL}: On"
        assert page._intents.snapshot().thaw().run_options['record_angular_support'] is True
        pill = _int_recording(qapp, page, toggle=True)
        assert not pill.isChecked() and pill.text() == f"{LABEL}: Off"
        assert 'record_angular_support' not in page._intents.snapshot().thaw().run_options
        assert view.plot_options_dialog.projection.show_angular_support == shown, "recording never changes display"
        pill = _int_recording(qapp, page, toggle=True)
        assert pill.isChecked() and pill.text() == f"{LABEL}: On"
    page = StitchPage(empty_host_services(_Status()))
    try:
        pill = _pill(page)
        assert page.advanced.isAncestorOf(pill), 'Stitch recording belongs in Advanced'
        from xdart.gui.widgets.controls_panel import SubsectionCard
        assert any(card.isAncestorOf(pill) and card.title.text() == 'Integrate 1D'
                   for card in page.advanced.findChildren(SubsectionCard))
        assert not pill.isChecked() and pill.text() == f"{LABEL}: Off"
        pill.click()
        qapp.processEvents()
        assert pill.isChecked() and pill.text() == f"{LABEL}: On"
        pill.click()
        qapp.processEvents()
        document = page_settings(page)
        assert document["fields"]["record_angular_support"] is False and pill.text() == f"{LABEL}: Off"
    finally:
        page.shutdown()
        page.deleteLater()
        qapp.processEvents()
    fresh = StitchPage(empty_host_services(_Status()))
    try:
        _pill(fresh).click()  # On first, so reopening the saved Off is a real restoration
        qapp.processEvents()
        assert _pill(fresh).isChecked() and _pill(fresh).text() == f"{LABEL}: On"
        apply_page_settings(fresh, document)
        qapp.processEvents()
        assert not _pill(fresh).isChecked() and _pill(fresh).text() == f"{LABEL}: Off", "reopened, it says Off"
    finally:
        fresh.shutdown()
        fresh.deleteLater()
        qapp.processEvents()
    # the shared owner's reconciliation, a busy lock included, keeps the state text with the value
    field = ControlFormField(SectionId.PROCESSING, LABEL, RECORD_ANGULAR_SUPPORT, value=True,
                             kind=ControlFieldKind.BOOL)
    row = PillRow([field])
    pill = _pill(row)
    assert pill.text() == f"{LABEL}: On"
    locked = replace(field, value=False, enabled=False, reason="Locked while a Run is in progress.")
    row.apply_fields({RECORD_ANGULAR_SUPPORT: locked})
    assert (pill.text(), pill.isChecked(), pill.isEnabled()) == (f"{LABEL}: Off", False, False)
    assert pill.toolTip() == "Locked while a Run is in progress."
    row.apply_fields({RECORD_ANGULAR_SUPPORT: field})
    assert (pill.text(), pill.isChecked(), pill.isEnabled()) == (f"{LABEL}: On", True, True)
    row.deleteLater()


def test_c2_show_angular_coverage_is_a_display_choice_applied_on_okay(qapp):
    view, dialog, commands = _options(qapp)
    box = dialog.show_angular_support
    assert box.text() == "Show angular coverage"
    tip = box.toolTip()
    assert "Okay" in tip and "Cancel" in tip and LABEL in tip, tip
    before = box.isChecked()
    commands.clear()
    box.click()
    dialog.cancel_button.click()
    qapp.processEvents()
    assert not [command for command in commands if "angular_support" in repr(command)], "Cancel applies nothing"
    view.options.click()
    qapp.processEvents()
    assert dialog.show_angular_support.isChecked() == before, "Cancel discarded the edit"
    dialog.show_angular_support.click()
    dialog.accept_button.click()
    qapp.processEvents()
    applied = [command for command in commands if "angular_support" in repr(command)]
    assert applied, "Okay applies the display choice"
    assert not [command for command in commands if "record_angular_support" in repr(command)], "display only"


def test_c3_2d_pixel_coverage_says_it_needs_a_stitch_intensity_map(qapp):
    _view, dialog, _commands = _options(qapp)
    box = dialog.show_coverage_holes
    group = box.parentWidget()
    assert isinstance(group, QtWidgets.QGroupBox) and group.title() == "2D pixel coverage (Stitch only)"
    assert box.text() == "Shade uncovered bins" and not box.isEnabled(), "an ordinary Int result has no pixel coverage"
    assert "Stitch" in box.toolTip() and "Show angular coverage" in box.toolTip(), box.toolTip()
    dialog.set_coverage_available(True)
    assert box.isEnabled() and "pixel" in box.toolTip() and "Show angular coverage" in box.toolTip(), box.toolTip()


def test_c4_detector_coordinate_grid_names_its_frame_and_says_why_not(qapp):
    _view, dialog, _commands = _options(qapp)
    labels = [label for label in dialog.findChildren(QtWidgets.QLabel) if label.text() == "Detector coordinate grid"]
    assert len(labels) == 1, "Plot Options names the Detector coordinate grid choice"
    combo = labels[0].parentWidget().findChild(QtWidgets.QComboBox)
    assert combo is dialog.coordinate_grid
    assert [combo.itemText(index) for index in range(combo.count())] == ["Off", "q", "χ", "q + χ"]
    dialog.set_grid_available(True, visible=True)
    tip = combo.toolTip()
    assert "raw detector" in tip and "Run" in tip and "saved" in tip, tip
    reason = "The coordinate grid requires a uniform, flat, undistorted detector."
    dialog.set_grid_available(False, reason, visible=True)
    assert not combo.isEnabled() and combo.toolTip().endswith(reason), combo.toolTip()
