"""Frozen H2–H9 through shared owners, actual Qt events and production windows."""
from dataclasses import replace

import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtTest, QtWidgets


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _f1(root, widget, qapp):
    root.show(); root.activateWindow(); widget.setFocus(); qapp.processEvents()
    QtTest.QTest.keyClick(widget, QtCore.Qt.Key.Key_F1)
    qapp.processEvents()


def _dispose(root, qapp):
    root.close(); root.deleteLater(); qapp.processEvents()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def test_f1_nearest_context_unknown_fallback_and_shared_help_window(qapp):
    from xdart.gui.widgets.context_help import bind_topic, owner_for
    from xdart.gui.widgets.controls_panel import SectionCard, FormRow
    from xdart.gui.widgets.read_only_text import ReadOnlyTextPane
    from xdart.gui.tabs.scattering.browser_view import BrowserView
    root = QtWidgets.QWidget(); root.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
    layout = QtWidgets.QVBoxLayout(root)
    card = SectionCard("Experiment", number="2", accent="experiment"); layout.addWidget(card)
    card.set_help_topic("calibration.multi-step")
    row = FormRow(label="Detector", path=("test",), value="stable")
    card.body_layout.addWidget(row); row.set_help_topic("geometry.orientation")
    commits = []; row.valueChanged.connect(lambda *args: commits.append(args))
    owner = owner_for(root)
    try:
        root.show(); root.activateWindow(); row.editor.setFocus(); qapp.processEvents()
        QtTest.QTest.keyClicks(row.editor, " draft")
        _f1(root, row.editor, qapp)
        window = owner.window
        assert window.topic_id == "geometry.orientation" and not commits
        assert isinstance(window.pane, ReadOnlyTextPane) and window.pane.isReadOnly()
        assert isinstance(window.browser, BrowserView) and not window.browser.menus
        assert isinstance(window.search_row, FormRow)
        # Illustrated help mounts additional full shared owners; no page-built editor.
        from xdart.gui.tabs.scattering.scientific_view import ScientificView
        owners = (window.pane, window.browser, *window.findChildren(FormRow),
                  *window.findChildren(ScientificView))
        for widget in window.findChildren(QtWidgets.QWidget):
            if isinstance(widget, (QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit,
                                   QtWidgets.QListWidget, QtWidgets.QLineEdit, QtWidgets.QComboBox)):
                ancestor = widget
                while ancestor is not None and ancestor not in owners:
                    ancestor = ancestor.parentWidget()
                assert ancestor is not None, type(widget)
        original = window.pane.toPlainText()
        window.pane.setFocus(); QtTest.QTest.keyClicks(window.pane, "replace")
        assert window.pane.toPlainText() == original
        window.pane.selectAll(); window.pane.copy()
        assert qapp.clipboard().text() == original
        for widget in (window.pane, window.browser.scans, window.browser.scans_label,
                       window.search_row, window.search_row.editor, card.help_button):
            assert widget.toolTip()
        _f1(root, row.editor, qapp)
        assert owner.window is window
        root.activateWindow(); row.editor.setFocus(); qapp.processEvents()
        QtTest.QTest.keyClicks(row.editor, " more")
        QtTest.QTest.mouseClick(card.help_button, QtCore.Qt.MouseButton.LeftButton)
        qapp.processEvents()
        assert not commits and "draft" in row.current_value()
        bind_topic(row, "unknown.β")
        _f1(root, row.editor, qapp)
        assert window.topic_id is None and "unknown.β" in window.pane.toPlainText()
        bind_topic(row, "")
        _f1(root, row.editor, qapp)
        assert window.topic_id == "calibration.multi-step"
        _f1(root, root, qapp)
        assert window.topic_id is None
        window.close(); owner.show_topic("analysis.plot-roi")
        assert owner.window.isVisible() and owner.window.topic_id == "analysis.plot-roi"
        root.hide(); qapp.processEvents()
        assert owner.window is None and owner.shortcut is None
        root.show(); owner.show_index(); assert owner.window.isVisible()
    finally:
        _dispose(root, qapp)


def test_real_search_selection_and_f1_inside_help_return_to_index(qapp):
    from xrd_tools.help_topics import TOPICS
    from xdart.gui.widgets.context_help import owner_for
    root = QtWidgets.QWidget(); owner = owner_for(root)
    try:
        root.show(); owner.show_index(); window = owner.window
        window.search_row.editor.setFocus()
        QtTest.QTest.keyClicks(window.search_row.editor, "polarization flat cutoff")
        qapp.processEvents()
        assert window.browser.scans.count() == 1
        item = window.browser.scans.item(0)
        QtTest.QTest.mouseClick(window.browser.scans.viewport(), QtCore.Qt.MouseButton.LeftButton,
                               pos=window.browser.scans.visualItemRect(item).center())
        qapp.processEvents()
        assert window.topic_id == "corrections.existing"
        _f1(window, window.pane, qapp)
        assert window.topic_id is None and window.search_row.editor.hasFocus()
        # declared oracle repair (PHASE-ANALYSIS-128): the whole index is the catalogue (six topics at 1.2.7)
        assert window.browser.scans.count() == len(TOPICS)
        QtTest.QTest.keyClicks(window.search_row.editor, "nothing_matches_123")
        assert window.browser.scans.count() == 0
        window.search_row.editor.clear(); assert window.browser.scans.count() == len(TOPICS)
    finally:
        _dispose(root, qapp)


def test_help_is_presentation_only_and_available_outside_editor_locks(qapp):
    from xdart.gui.widgets.controls_panel import ControlsPanel, FormRow
    from xrd_tools.session.readiness import ControlAction, ControlFormField, ControlFieldKind, SectionId
    from xdart.gui.widgets.context_help import owner_for
    panel = ControlsPanel(); emitted = []; panel.controlActionRequested.connect(emitted.append)
    row = FormRow(label="Flat", path=("corrections", "flat"), value="/data/flat.tif", browse=True)
    row.set_purpose_tooltip("Divide by this detector flat.")
    row.set_help_topic("corrections.existing")
    panel.processing_card.body_layout.addWidget(row)
    try:
        panel.show(); panel.set_editors_enabled(False)
        for card, topic in ((panel.experiment_card, "geometry.orientation"),
                            (panel.processing_card, "corrections.existing")):
            assert card.help_button.isEnabled() and card.help_button.spec.action is ControlAction.HELP
            card.help_button.click()
            assert owner_for(panel).window.topic_id == topic
        assert not emitted and row.current_value() == "/data/flat.tif"
        panel.set_editors_enabled(True)
        assert row.toolTip() == "/data/flat.tif"
        row.apply_field(ControlFormField(SectionId.PROCESSING, "Flat", row.path, "/data/flat.tif", ControlFieldKind.LINE,
                                        enabled=False, reason="Locked during operation"))
        assert row.toolTip() == "Locked during operation"
    finally:
        _dispose(panel, qapp)


def test_actual_main_menus_move_one_index_action_and_page_contexts(qapp):
    from xdart._gui_main import Main
    from xdart.gui.pages.values import SCATTERING_PAGE_KEY, STITCH_TOOL_KEY, RSM_TOOL_KEY
    from xdart.gui.widgets.context_help import owner_for
    from xdart.gui.widgets.controls_panel import SubsectionCard
    main = Main(selected_page_key=SCATTERING_PAGE_KEY)
    try:
        owner = owner_for(main); action = main.actionHelpIndex
        assert action.shortcut().isEmpty()
        for key, topic in ((SCATTERING_PAGE_KEY, "geometry.orientation"),
                           (STITCH_TOOL_KEY, "calibration.multi-step"),
                           (RSM_TOOL_KEY, "calibration.multi-step")):
            main.select_page(key); qapp.processEvents()
            menu = main._attached_help_menu
            assert menu.actions()[0] is action and menu.actions().count(action) == 1
            assert not any(a.text() == "Help" for a in menu.actions())
            action.trigger(); window = owner.window; assert window.topic_id is None
            page = main.page_handle.widget
            controls = page._shell.controls if key == SCATTERING_PAGE_KEY else page.controls
            controls.experiment_card.help_button.click()
            assert owner.window is window and window.topic_id == topic
            controls.processing_card.help_button.click()
            assert window.topic_id == "corrections.existing"
            controls.experiment_card.show()  # the empty Project otherwise hides this bound control
            _f1(main, controls.experiment_card.help_button, qapp)
            assert window.topic_id == topic
            if key != SCATTERING_PAGE_KEY:
                advanced = main.page_handle.widget.advanced
                advanced.show(); qapp.processEvents()
                group = next(c for c in advanced.findChildren(SubsectionCard)
                             if c.property("helpTopic") == "corrections.existing")
                advanced.set_editors_enabled(False)
                assert not group.body.isEnabled() and group.help_button.isEnabled()
                group.help_button.click()
                assert owner_for(advanced).window.topic_id == "corrections.existing"
                _f1(advanced, group.help_button, qapp)
                assert owner_for(advanced).window.topic_id == "corrections.existing"
        main.close(); qapp.processEvents()
        assert owner.window is None and owner.shortcut is None
    finally:
        _dispose(main, qapp)


def test_calibration_contexts_literal_failure_and_clean_shutdown(tmp_path, qapp):
    from xdart.gui.tabs.calibration.dialog import CalibrationDialog, CalibrationContext
    from xdart.gui.widgets.context_help import owner_for
    from xdart.gui.pages.values import PageCleanup
    dialog = CalibrationDialog(CalibrationContext("Stitch", tmp_path)); owner = owner_for(dialog)
    try:
        dialog.show(); qapp.processEvents()
        before = tuple((k, row.current_value()) for k, row in dialog._formrows.items())
        dialog.tabs.cornerWidget().click(); assert owner.window.topic_id == "calibration.multi-step"
        for tab, topic in (("instrument", "geometry.orientation"), ("rough", "geometry.orientation"),
                           ("select", "calibration.frame-choice"), ("calibrate", "calibration.diagnostics")):
            dialog.tabs.setCurrentWidget(dialog._tab_pages[tab])
            widget = next(w for w in dialog._tab_pages[tab].findChildren(QtWidgets.QWidget)
                          if w.focusPolicy() != QtCore.Qt.FocusPolicy.NoFocus and w.isVisible())
            _f1(dialog, widget, qapp); assert owner.window.topic_id == topic
        message = "uncoded failure β\nexact second line"
        dialog.status.setText(message); dialog.fit_summary.set_text("RMS unchanged\n" + message)
        dialog.tabs.setCurrentWidget(dialog._tab_pages["calibrate"])
        _f1(dialog, dialog.fit_summary, qapp)
        assert message in owner.window.pane.toPlainText()
        assert before == tuple((k, row.current_value()) for k, row in dialog._formrows.items())
        assert not dialog.active()
        assert dialog.shutdown().status is PageCleanup.CLEAN
        assert owner.window is None and owner.shortcut is None
    finally:
        _dispose(dialog, qapp)


def test_roi_result_warnings_survive_redraw_stale_and_refusal_then_retire(tmp_path, qapp):
    from xdart.gui.widgets.context_help import owner_for
    from tests.xdart.scattering.test_roi_combined_window import _open, _settle
    from tests.xdart.scattering.test_p37_analysis_mount import _page, _close_page, _roi_navigation_table
    _roi_navigation_table(tmp_path, missing_label=7); page = _page()
    try:
        dialog, picker = _open(page, tmp_path / "navigation.nexus", qapp)
        picker.compute_btn.click(); _settle(page, qapp)
        result = replace(dialog._vnext_roi_result, diagnostics=("ROI_INVALID_MASK_IGNORED", "literal β\nline 2"))
        dialog.set_vnext_roi_result(result); _settle(page, qapp)
        assert result.code == "OK" and dialog._vnext_roi_result is result
        assert "!" in dialog.help_button.text()
        dialog.help_button.click(); window = owner_for(dialog).window
        assert "OK\nROI_INVALID_MASK_IGNORED\nliteral β\nline 2" in window.pane.toPlainText()
        dialog.refresh_roi_cursor(); picker.reducer_combo.setCurrentText("sum")
        dialog.help_button.click()
        assert "retained" in window.pane.toPlainText().lower() and "stale" in window.pane.toPlainText().lower()
        assert "literal β\nline 2" in window.pane.toPlainText() and "Out of date" in dialog.roi_caption.text()
        picker.scientific.keyed_panel(picker._image_key).selector.setCurrentIndex(1)
        _settle(page, qapp)  # actual refused raw preview, adopted through the page's terminal projection
        dialog.help_button.click()
        assert "ROI_RAW_UNAVAILABLE" in window.pane.toPlainText() and "!" in dialog.help_button.text()
        dialog.set_help_diagnostic("ROI_PREVIEW_LIMIT_EXCEEDED", ("literal refusal",))
        _f1(dialog, dialog.x_combo, qapp)
        assert "ROI_PREVIEW_LIMIT_EXCEEDED\nliteral refusal" in window.pane.toPlainText()
        assert "64 MiB" in window.pane.toPlainText() and "!" in dialog.help_button.text()
        dialog.set_vnext_metadata(dialog._vnext_table_result)
        dialog.help_button.click()
        assert "literal β" not in window.pane.toPlainText() and "literal refusal" not in window.pane.toPlainText()
        owner = owner_for(dialog); dialog.close(); assert owner.window is None and owner.shortcut is None
    finally:
        _close_page(page, qapp)


def test_calibration_busy_close_retains_help_until_worker_cleanup(tmp_path, qapp):
    """H9/A7: a real worker's admitted operation and real ignored Qt Close event."""
    from threading import Event
    from xdart.gui.pages.operation_owner import SingleWorkerOwner
    from xdart.gui.pages.values import PageCleanup
    from xdart.gui.tabs.calibration.dialog import CalibrationDialog, CalibrationContext
    from xdart.gui.widgets.context_help import owner_for
    entered, release = Event(), Event()

    def held_job(request, cancel, progress):
        entered.set()
        assert release.wait(5)
        return request

    dialog = CalibrationDialog(CalibrationContext("RSM", tmp_path))
    # The science runner is outside the help seam; the actual operation/close owner is exercised.
    dialog._owner = SingleWorkerOwner(held_job)
    dialog._start("held", ())
    assert entered.wait(2)
    owner = owner_for(dialog)
    try:
        dialog.show(); owner.show_topic("calibration.diagnostics")
        identity = dialog._owner.current_identity; window = owner.window
        assert not dialog.tabs.tabBar().isEnabled() and dialog.tabs.cornerWidget().isEnabled()
        dialog.tabs.cornerWidget().click()
        assert dialog._owner.current_identity is identity and dialog.active()
        event = QtGui.QCloseEvent(); qapp.sendEvent(dialog, event)
        assert not event.isAccepted() and owner.window is window and window.isVisible()
        assert owner.shortcut is not None
        release.set()
        dialog._owner._worker.join(2)
        assert dialog.shutdown().status is PageCleanup.CLEAN
        assert owner.window is None and owner.shortcut is None
    finally:
        release.set()
        if dialog._owner._worker is not None:
            dialog._owner._worker.join(2)
        dialog.shutdown(); _dispose(dialog, qapp)


@pytest.mark.parametrize("return_to_root", ("dismiss", "reactivate"))
def test_real_int_help_visit_reconciles_draft_before_calibrate(tmp_path, qapp, monkeypatch, return_to_root):
    """B1/H4/A2: opening help commits nothing; the next real action captures the display."""
    from xdart._gui_main import Main
    from xdart.gui.pages.values import SCATTERING_PAGE_KEY
    from xdart.gui.widgets.context_help import owner_for
    from xdart.gui.widgets.controls_panel import FormRow
    from xdart.gui.tabs.scattering.controls_inventory import INT_1D_POINTS, integration_values
    from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
    main = Main(selected_page_key=SCATTERING_PAGE_KEY)
    page = main.page_handle.widget
    captures = []
    monkeypatch.setattr("xdart.gui.tabs.scattering.page.resolve_calibration_executable", lambda: "/unused/pyFAI-calib2")
    # Cancel the actual chooser boundary after it sees the actual revisioned intent.
    def chooser(*_args):
        captures.append(page._intents.snapshot())
        return ""
    page._authoring_source_chooser = chooser
    try:
        snapshot = page._intents.snapshot(); intent = snapshot.thaw(); intent.project_root = str(tmp_path)
        page._intents.commit(intent, expected_revision=snapshot.revision); page._refresh_shell()
        controls = page._shell.controls
        row = next(r for r in controls.findChildren(FormRow) if r.path == INT_1D_POINTS)
        main.show(); main.activateWindow(); row.editor.setFocus(); qapp.processEvents()
        assert row.editor.hasFocus()
        before = page._intents.snapshot()
        row.editor.selectAll(); QtTest.QTest.keyClicks(row.editor, "444")
        QtTest.QTest.mouseClick(controls.processing_card.help_button, QtCore.Qt.MouseButton.LeftButton)
        qapp.processEvents()
        owner = owner_for(main)
        assert owner.window.isVisible() and page._intents.snapshot().revision == before.revision
        if return_to_root == "dismiss":
            owner.window.close()
        main.activateWindow(); qapp.processEvents()  # no manual editor refocus
        print("B1 Int return", return_to_root, "hold", row._help_focus_transfer,
              "focus", main.focusWidget(), "display", row.current_value())
        page._shell.commandRequested.emit(ShellCommand(ShellCommandKind.CONTROL_ACTION, "calibrate"))
        assert len(captures) == 1
        assert integration_values(captures[0].thaw())[INT_1D_POINTS] == int(row.current_value()) == 444
        assert not row._help_focus_transfer
    finally:
        _dispose(main, qapp)


@pytest.mark.parametrize("dismiss", (False, True))
def test_real_rsm_authoritative_update_replaces_help_draft(tmp_path, qapp, dismiss):
    """B1: RSM's real model projection must replace a held presentation draft."""
    from xdart._gui_main import Main
    from xdart.gui.pages.values import RSM_TOOL_KEY
    from xdart.gui.widgets.context_help import owner_for
    main = Main(selected_page_key=RSM_TOOL_KEY); page = main.page_handle.widget
    try:
        page.set_field("project_root", str(tmp_path)); page.set_field("adv_rows", "333"); page.advanced.show()
        row = page._rows["adv_rows"]
        page.advanced.activateWindow(); row.editor.setFocus(); qapp.processEvents()
        assert row.editor.hasFocus()
        row.editor.selectAll(); QtTest.QTest.keyClicks(row.editor, "444")
        _f1(page.advanced, row.editor, qapp)
        owner = owner_for(page.advanced)
        if dismiss:
            owner.window.close(); page.advanced.activateWindow(); qapp.processEvents()
        page.set_field("adv_rows", "555")
        print("B1 RSM update", dismiss, "hold", row._help_focus_transfer,
              "model", page._fields["adv_rows"].value, "display", row.current_value())
        assert page._fields["adv_rows"].value == row.current_value() == row.editor.text() == "555"
        assert not row._help_focus_transfer
    finally:
        _dispose(main, qapp)


def test_real_stitch_busy_to_idle_keeps_advanced_help_available(tmp_path, qapp):
    """T1/A2: exercise the real operation owner and page's Advanced lock projection."""
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.tabs.stitch.page import StitchPage
    from xdart.gui.widgets.context_help import owner_for
    from xdart.gui.widgets.controls_panel import SubsectionCard
    from tests.xdart.pages.test_stitch_page import _Status, _wait
    from tests.xdart.pages.test_stitch_page_options import _close
    from tests.xdart.tools.test_stitch_native_values import _native_form
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(_native_form(tmp_path, "multigeometry", scans="1"))
        page.advanced.show(); qapp.processEvents()
        group = next(c for c in page.advanced.findChildren(SubsectionCard)
                     if c.property("helpTopic") == "corrections.existing")
        assert group.body.isEnabled() and group.help_button.isEnabled()
        page.begin_preview()
        assert page._owner.busy and not group.body.isEnabled() and group.help_button.isEnabled()
        group.help_button.click(); assert owner_for(page.advanced).window.topic_id == "corrections.existing"
        _wait(page, qapp)
        assert not page._owner.busy and group.body.isEnabled() and group.help_button.isEnabled()
        page.preview_dialog.close()
    finally:
        _close(page, qapp)
