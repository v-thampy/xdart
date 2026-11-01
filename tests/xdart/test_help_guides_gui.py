"""NEXT-128-HELP-GUIDES G3: each tool window's top-level "?" opens its illustrated physics guide in the browser, and
the Help window offers the same guide for the tool's topics (the maintainer, 2026-10-31: "These would be the top level
help that open the question mark at the top of the page"; "System browser (Recommended)"). Only the operating system's
browser call is answered, as the file choosers are. Frozen before the product change (red on d923716b).
"""
from pathlib import Path

import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def opened(monkeypatch):
    calls = []
    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl", staticmethod(lambda url: calls.append(url) or True))
    return calls


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    monkeypatch.setenv("XDART_PHASE_LIBRARY", str(tmp_path / "phase-library"))


def _dialogs():
    from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    return ((PeakFitDialog(vnext_submit=lambda *_args: None), "peak-fit.html"),
            (PhaseFitDialog(vnext_submit=lambda *_args: None), "phase-analysis.html"))


def test_g3_each_tool_windows_top_question_mark_opens_its_guide(qapp, opened):
    from xdart.gui.widgets.controls_panel import ActionButton
    for dialog, name in _dialogs():
        try:
            dialog.show()
            qapp.processEvents()
            button = dialog.guide_button
            assert isinstance(button, ActionButton) and button.text() == "?" and button.toolTip(), name
            top = button.mapTo(dialog, QtCore.QPoint(0, 0)).y()
            assert top < dialog.fit_view.mapTo(dialog, QtCore.QPoint(0, 0)).y(), name  # at the top of the window
            before = len(opened)
            button.click()
            qapp.processEvents()
            assert len(opened) == before + 1
            url = opened[-1]
            assert url.isLocalFile() and Path(url.toLocalFile()).name == name and Path(url.toLocalFile()).is_file()
        finally:
            dialog.close()


def test_g3_the_help_window_offers_the_guide_only_for_topics_that_have_one(qapp, opened):
    from xdart.gui.widgets.context_help import owner_for
    root = QtWidgets.QWidget()
    try:
        owner = owner_for(root)
        owner.show_topic("peak-fit.background")
        window = owner.window
        assert not window.guide_button.isHidden() and window.guide_button.toolTip()
        window.guide_button.click()
        qapp.processEvents()
        assert opened and Path(opened[-1].toLocalFile()).name == "peak-fit.html"
        owner.show_topic("calibration.multi-step")
        assert window.guide_button.isHidden()
    finally:
        owner_for(root).shutdown()
        root.deleteLater()


def test_g3_a_missing_guide_is_named_and_nothing_opens(qapp, opened, monkeypatch, tmp_path):
    from xrd_tools import help_topics
    from xdart.gui.widgets.context_help import owner_for
    monkeypatch.setattr(help_topics, "guide_path", lambda identifier: tmp_path / "gone.html")
    dialog, _name = _dialogs()[0]
    try:
        dialog.show()
        qapp.processEvents()
        dialog.guide_button.click()
        qapp.processEvents()
        assert not opened
        window = owner_for(dialog).window
        assert window is not None and "missing" in window.pane.toPlainText().casefold()
    finally:
        dialog.close()
