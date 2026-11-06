"""NEXT-128-HELP-GUIDES G3: the Help window offers each tool's illustrated physics guide for the tool's topics, and opens
it in the system browser ("System browser (Recommended)"). Only the operating system's browser call is answered, as the
file choosers are. Frozen before the product change (red on d923716b).

NEXT-128-WINDOW-POLISH (TQ) retired the tool windows' top "?" (the maintainer, 2026-11-02: "the question mark at the very
top of the windows that brings up the illustrated guide is not really required. the same can be gotten to by clicking the
illustrated guide button in any of the help windows, right?", then "Remove it (Recommended)"). Its rows are replaced:
neither window has it, every section "?" reaches the guide through the Help window's Illustrated guide, F1 in an option
still opens help before any help was opened, and the missing-guide and browser-refusal rows go through the Help window.
Frozen before that change (red on 034723d4).

NEXT-129-SOURCE-GUIDE (the maintainer, 2026-11-05, on 1.2.8: "the button that brought up the illustrated guides is no
longer there in the help windows"): the Source card's "?" opened "fit.source", the topic both fit windows share, which no
one tool's guide owned, so the Help window hid Illustrated guide there. The windows are now built as Int's page builds
them (``source=True``), so the every-"?" row reaches the Source card's three "?" (a disclosed oracle repair: that row
built them without the card, and it checked the topic's own guide, which a shared topic has none of, where it now
checks the guide that opens). The shared topic offers the guide of the window it was opened from, by "?", by F1 and from
the Help window's topic list; a topic another tool owns keeps its own guide, a topic of no tool offers none, and a
window that is not a tool's offers none for the shared topic. Frozen before the product change (red on 64466046).
"""
from pathlib import Path

import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtTest, QtWidgets

KINDS = ("peak", "phase")
GUIDES = {"peak": "peak-fit.html", "phase": "phase-analysis.html"}


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


def _dialog(kind, source=False):
    from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    return (PeakFitDialog if kind == "peak" else PhaseFitDialog)(vnext_submit=lambda *_args: None, source=source)


def _opens(window, opened, qapp, name):
    """Illustrated guide is offered and opens the packaged guide ``name`` in the system browser."""
    assert not window.guide_button.isHidden() and window.guide_button.toolTip(), window.topic_id
    before = len(opened)
    window.guide_button.click()
    qapp.processEvents()
    assert len(opened) == before + 1, window.topic_id
    path = Path(opened[-1].toLocalFile())
    assert opened[-1].isLocalFile() and path.name == name and path.is_file(), (window.topic_id, path)


def _pick(window, qapp, identifier):
    """Select a topic in the Help window's own topic list as a user does: search for it, then click it (the pattern
    test_context_help runs on CI)."""
    search = window.search_row.editor
    search.clear()
    QtTest.QTest.keyClicks(search, identifier)
    qapp.processEvents()
    scans = window.browser.scans
    assert scans.count() == 1, identifier
    rect = scans.visualItemRect(scans.item(0))
    assert scans.viewport().rect().contains(rect.center()), identifier
    QtTest.QTest.mouseClick(scans.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=rect.center())
    qapp.processEvents()
    assert window.topic_id == identifier


@pytest.mark.parametrize("kind", KINDS)
def test_tq_neither_tool_window_has_a_top_question_mark(qapp, kind):
    from xdart.gui.widgets.controls_panel import ActionButton
    dialog = _dialog(kind)
    try:
        dialog.show()
        qapp.processEvents()
        assert not hasattr(dialog, "guide_button") and not hasattr(ActionButton, "for_guide")
        assert dialog.layout().itemAt(0).widget() is dialog.split, "no row above the options and plots"
        tips = [button.toolTip().casefold() for button in dialog.findChildren(ActionButton)]
        assert not any("illustrated physics guide" in tip for tip in tips)
    finally:
        dialog.close()


@pytest.mark.parametrize("kind", KINDS)
def test_tq_every_section_question_mark_offers_its_tools_guide_in_the_help_window(qapp, opened, kind):
    from xdart.gui.widgets.context_help import owner_for
    from xdart.gui.widgets.controls_panel import ActionButton
    dialog = _dialog(kind, source=True)  # as Int's page builds it: the Source card first (NEXT-129-SOURCE-GUIDE)
    try:
        dialog.show()
        qapp.processEvents()
        buttons = [button for button in dialog.findChildren(ActionButton) if button.text() == "?"]
        assert buttons
        topics = set()
        for button in buttons:
            button.click()
            qapp.processEvents()
            window = owner_for(button).window  # Advanced cards have their own top-level help owner.
            assert window is not None and window.isVisible()
            topics.add(window.topic_id)
            _opens(window, opened, qapp, GUIDES[kind])
        assert "fit.source" in topics and len(topics) > 1, topics
    finally:
        dialog.close()


@pytest.mark.parametrize("kind", KINDS)
def test_sg_f1_and_the_topic_list_offer_the_windows_guide_for_the_shared_source_topic(qapp, opened, kind):
    dialog = _dialog(kind, source=True)
    try:
        dialog.show()
        dialog.activateWindow()
        dialog.source_row.choose_btn.setFocus()
        qapp.processEvents()
        QtTest.QTest.keyClick(dialog.source_row.choose_btn, QtCore.Qt.Key.Key_F1)
        qapp.processEvents()
        owner = getattr(dialog, "_context_help_owner", None)
        assert owner is not None and owner.window is not None
        window = owner.window
        assert window.topic_id == "fit.source"
        _opens(window, opened, qapp, GUIDES[kind])
        # from the Help window's own list: a topic another tool owns keeps its tool's guide, a topic of no tool has
        # none, and the shared topic again offers this window's
        other = "phase" if kind == "peak" else "peak"
        _pick(window, qapp, "phase-analysis.phases" if other == "phase" else "peak-fit.weighting")
        _opens(window, opened, qapp, GUIDES[other])
        _pick(window, qapp, "calibration.multi-step")
        assert window.guide_button.isHidden()
        _pick(window, qapp, "fit.source")
        _opens(window, opened, qapp, GUIDES[kind])
    finally:
        dialog.close()


def test_sg_the_shared_source_topic_offers_no_guide_outside_a_tool_window(qapp, opened):
    from xdart.gui.widgets.context_help import owner_for
    root = QtWidgets.QWidget()
    try:
        owner = owner_for(root)
        owner.show_topic("fit.source")
        window = owner.window
        assert window.topic_id == "fit.source" and window.guide_button.isHidden()
        assert not opened
    finally:
        owner_for(root).shutdown()
        root.deleteLater()


@pytest.mark.parametrize("kind", KINDS)
def test_ig_f1_with_no_option_focused_opens_an_index_offering_the_windows_guide(qapp, opened, kind):
    """NEXT-129-SMALL-FIXES item 9 (the maintainer's "Add it (Recommended)"; the gap NEXT-129-SOURCE-GUIDE left): in a
    tool window, F1 with no option focused opens the help index, which offered no illustrated guide. It now offers the
    window's own (its ``help_guide_topic``) and opens it; the Help window's own F1 returns to that index."""
    dialog = _dialog(kind, source=True)
    try:
        dialog.show()
        dialog.activateWindow()
        dialog.split.controls.setFocus()  # the options column itself: no option, no topic
        qapp.processEvents()
        QtTest.QTest.keyClick(dialog.split.controls, QtCore.Qt.Key.Key_F1)
        qapp.processEvents()
        owner = getattr(dialog, "_context_help_owner", None)
        assert owner is not None and owner.window is not None
        window = owner.window
        assert window.topic_id is None, "F1 with no option focused opens the index"
        _opens(window, opened, qapp, GUIDES[kind])
        _pick(window, qapp, "calibration.multi-step")  # a topic of no tool: no guide
        assert window.guide_button.isHidden()
        QtTest.QTest.keyClick(window.pane, QtCore.Qt.Key.Key_F1)  # the Help window's own F1: the index again
        qapp.processEvents()
        assert window.topic_id is None
        _opens(window, opened, qapp, GUIDES[kind])
    finally:
        dialog.close()


def test_ig_the_index_offers_no_guide_outside_a_tool_window(qapp, opened):
    """The negative control: a root that names no ``help_guide_topic`` (a plain widget, as Int's main window) has an index
    with no Illustrated guide."""
    from xdart.gui.widgets.context_help import owner_for
    root = QtWidgets.QWidget()
    try:
        owner = owner_for(root)
        owner.show_index()
        window = owner.window
        assert window.topic_id is None and window.guide_button.isHidden()
        assert not opened
    finally:
        owner_for(root).shutdown()
        root.deleteLater()


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
    dialog = _dialog("peak")
    try:
        dialog.show()
        qapp.processEvents()
        dialog.range_row.help_button.click()
        qapp.processEvents()
        window = owner_for(dialog).window
        window.guide_button.click()
        qapp.processEvents()
        assert not opened
        assert "missing" in window.pane.toPlainText().casefold()
    finally:
        dialog.close()


@pytest.mark.parametrize("kind", KINDS)
def test_tq_f1_in_an_option_opens_its_topic_before_any_help_was_opened(qapp, opened, kind):
    dialog = _dialog(kind)
    try:
        dialog.show()
        dialog.activateWindow()
        editor, expected = ((dialog.model_combo, "peak-fit.model") if kind == "peak" else
                            (dialog.profile_combo, "phase-analysis.peak-shape"))
        editor.setFocus()
        qapp.processEvents()
        QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_F1)
        qapp.processEvents()
        owner = getattr(dialog, "_context_help_owner", None)
        assert owner is not None and owner.window is not None, "F1 is bound when the window opens"
        assert owner.window.topic_id == expected and not opened
    finally:
        dialog.close()


@pytest.mark.parametrize("kind", KINDS)
def test_a_browser_refusal_names_the_reason_in_the_help_window(qapp, monkeypatch, kind):
    from xdart.gui.widgets.context_help import owner_for
    from xdart.gui.widgets.controls_panel import ActionButton
    calls = []
    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl", staticmethod(lambda url: calls.append(url) or False))
    dialog = _dialog(kind)
    try:
        dialog.show()
        qapp.processEvents()
        next(button for button in dialog.findChildren(ActionButton) if button.text() == "?").click()
        qapp.processEvents()
        window = owner_for(dialog).window
        assert window is not None and window.isVisible()
        window.guide_button.click()
        qapp.processEvents()
        assert len(calls) == 1 and calls[0].isLocalFile()
        assert "system browser did not open" in window.pane.toPlainText().casefold()
        assert not window.guide_button.autoDefault()
        search, button = window.search_row.geometry(), window.guide_button.geometry()
        assert max(search.top(), button.top()) < min(search.bottom(), button.bottom())
        assert search.right() < button.left() and button.width() > 0
        before = len(calls)
        window.search_row.editor.setFocus()
        QtTest.QTest.keyClick(window.search_row.editor, QtCore.Qt.Key.Key_Return)
        qapp.processEvents()
        assert len(calls) == before
        dialog.hide()
        qapp.processEvents()
        assert owner_for(dialog).window is None
    finally:
        dialog.close()


def test_f7_the_help_window_offers_the_strain_guide(qapp, opened):
    """NEXT-128-HELP-FOLLOWUPS F7: until the Strain window exists, its guide opens from the strain topic."""
    from xdart.gui.widgets.context_help import owner_for
    root = QtWidgets.QWidget()
    try:
        owner = owner_for(root)
        owner.show_topic("strain.sin2psi")
        window = owner.window
        assert window.topic_id == "strain.sin2psi" and not window.guide_button.isHidden()
        window.guide_button.click()
        qapp.processEvents()
        assert opened and Path(opened[-1].toLocalFile()).name == "strain.html"
        assert Path(opened[-1].toLocalFile()).is_file()
    finally:
        owner_for(root).shutdown()
        root.deleteLater()
