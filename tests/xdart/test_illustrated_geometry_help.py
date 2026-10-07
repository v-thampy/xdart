"""Illustrated help through the actual root/window and an exhaustive child census."""
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_headless_camera_labels_match_camera_editor(qapp):
    from xrd_tools.geometry_figures import camera_candidates
    from xdart.gui.widgets.camera_editor import CameraEditor, mounting_description
    editor = CameraEditor()
    try:
        for candidate in camera_candidates():
            editor.set_value(candidate.camera)
            assert candidate.label == mounting_description(editor.value())
            assert editor.value() == candidate.camera
    finally:
        editor.deleteLater(); qapp.processEvents()


def test_help_figures_are_lazy_shared_and_source_free(qapp, monkeypatch):
    from xdart.gui.widgets.context_help import owner_for
    from xdart.gui.tabs.scattering.scientific_view import ScientificView
    from xdart.gui.tabs.scattering.browser_view import BrowserView
    from xdart.gui.tabs.scattering.results_table import ResultsTable
    from xdart.gui.widgets.controls_panel import FormRow, ActionButton
    from xdart.gui.widgets.read_only_text import ReadOnlyTextPane
    from xrd_tools.help_topics import TOPICS
    import xrd_tools.io
    from xdart.gui.tabs.calibration.dialog import _run_job
    before = {t.identifier: t.text for t in TOPICS}
    def forbidden(*args, **kwargs):
        pytest.fail('Offline help invoked a source or worker boundary.')
    # The actual source boundary and job dispatch are tripwires, not replaced renderers.
    monkeypatch.setattr('xdart.gui.tabs.calibration.dialog._run_job', forbidden)
    monkeypatch.setattr(xrd_tools.io, 'read_image', forbidden)
    root = QtWidgets.QWidget(); owner = owner_for(root)
    try:
        root.show(); owner.show_index(); window = owner.window
        assert not window.findChildren(ScientificView)
        owner.show_topic('geometry.orientation')
        views = window.findChildren(ScientificView)
        assert len(views) == 1 and window.figure_view is views[0]
        window.show_orientation_check()
        assert window.step_row.editor.count() == 4
        for step in range(4):
            window.step_row.editor.setCurrentIndex(step); qapp.processEvents()
            assert len(window.findChildren(ScientificView)) == 1
            assert window.figure_view.keyed_projection and window.pane.toPlainText()
        for topic in TOPICS:
            window.show_topic(topic.identifier)
            assert topic.text in window.pane.toPlainText()
            assert len(window.findChildren(ScientificView)) == 1
        owners = tuple(o for cls in (ScientificView, BrowserView, ResultsTable, FormRow,
                                     ActionButton, ReadOnlyTextPane) for o in window.findChildren(cls))
        # Every display/editor/control, including newly added ones, is inside its full owner.
        for widget in window.findChildren(QtWidgets.QWidget):
            if isinstance(widget, (QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit,
                       QtWidgets.QAbstractItemView, QtWidgets.QLineEdit, QtWidgets.QComboBox,
                       QtWidgets.QAbstractButton)) or widget.inherits('PlotWidget'):
                assert any(widget is o or o.isAncestorOf(widget) for o in owners), type(widget)
        assert all(w.toolTip() for w in (window.step_row, window.step_row.editor,
                                        window.figure_row, window.figure_row.editor, window.run_button))
        assert {t.identifier: t.text for t in TOPICS} == before
        window.pane.setFocus(); QtTest.QTest.keyClick(window.pane, QtCore.Qt.Key.Key_F1)
        qapp.processEvents(); assert window.topic_id is None
    finally:
        root.close(); root.deleteLater(); qapp.processEvents()


def test_run_on_my_frames_resolves_only_the_calibration_root_capability(qapp, tmp_path):
    from xdart.gui.widgets.context_help import owner_for
    from xdart.gui.tabs.calibration.dialog import CalibrationDialog, CalibrationContext
    from xdart._gui_main import Main
    roots = (QtWidgets.QWidget(), Main(), CalibrationDialog(CalibrationContext('Stitch', tmp_path)))
    try:
        for root in roots:
            root.show(); owner_for(root).show_topic('geometry.orientation')
            window = owner_for(root).window
            assert window.run_button.isEnabled() == isinstance(root, CalibrationDialog)
            assert 'calibration' in window.run_button.toolTip().lower()
            if isinstance(root, CalibrationDialog):
                window.run_button.click(); qapp.processEvents()
                assert root.orientation_panel.isVisible() and not root.active()
            else:
                assert not root.findChildren(type(roots[-1]))
    finally:
        for root in reversed(roots):
            root.close(); root.deleteLater()
        qapp.processEvents()
