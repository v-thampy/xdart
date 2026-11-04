"""NEXT-128-WINDOW-POLISH item 14, the fit windows' frame selector (the maintainer's live check, 2026-11-03: "we also
want a frame selection widget like in the main window to be able to go to different frames"; his "Mirror Int's frames
(Recommended)"): the main window's ``<`` / frame / ``>`` lifted into one shared ``FrameNavigator`` (hard rule 14: Int's
footer and keyed image panels mount it, and so does the Source card's Frame row). Following Int's display, it lists
Int's frames and a step moves the main window; this window then shows that frame's trace. With a chosen NeXus file it
steps that file's frames and Int stays put. A running Batch disables it."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtWidgets

from tests.core.test_int_fit_source import Q, _peak, _result, _stored
from tests.xdart.scattering.test_fit_tools_preservation import _close_page, _page


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _until(qapp, predicate, seconds=30.0):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    return predicate()


def _selector(count, current, keys=None):
    from xdart.gui.tabs.scattering.shell_widgets import CompactFrameSelector
    selector = CompactFrameSelector()
    keys = keys if keys is not None else [object() for _ in range(count)]
    for index, key in enumerate(keys):
        selector.add_frame(f"f{index}", key, f"scan:{index}")
    selector.setCurrentIndex(current)
    return selector, keys


def test_n1_the_shared_navigator_steps_mirrors_and_lists_labels(qapp):
    from xdart.gui.tabs.scattering.shell_widgets import FrameNavigator
    navigator = FrameNavigator(track_ends=True)
    assert [widget.objectName() for widget in navigator.widgets] == [
        "e4FrameNavigationButton", "e3FrameSelector", "e4FrameNavigationButton"]
    assert (navigator.previous.text(), navigator.next.text()) == ("<", ">")
    assert all(widget.toolTip() for widget in navigator.widgets)
    seen = []
    navigator.selector.currentIndexChanged.connect(seen.append)
    source, keys = _selector(4, 2)
    navigator.mirror(source)
    assert [navigator.selector.itemText(index) for index in range(4)] == ["f0", "f1", "f2", "f3"]
    assert navigator.selector.currentIndex() == 2 and navigator.selector.itemData(1) is keys[1] and seen == []
    assert navigator.selector.itemData(1, 3) == "scan:1"  # ToolTipRole
    # a grown list adds only its new frames (Int's prefix rule), the selection follows, nothing is emitted
    keys.append(object())
    source.add_frame("f4", keys[4], "scan:4")
    source.setCurrentIndex(4)
    navigator.mirror(source)
    assert navigator.selector.count() == 5 and navigator.selector.currentIndex() == 4 and seen == []
    assert navigator.previous.isEnabled() and not navigator.next.isEnabled()  # at the last frame
    navigator.step(-1)
    assert navigator.selector.currentIndex() == 3 and seen == [3] and navigator.next.isEnabled()
    other, _ = _selector(2, 0)  # another list: replaced whole
    navigator.mirror(other)
    assert navigator.selector.count() == 2 and navigator.selector.currentIndex() == 0 and seen == [3]
    assert not navigator.previous.isEnabled() and navigator.next.isEnabled()
    navigator.set_labels((10, 11, 12), current=12)
    assert [navigator.selector.itemText(index) for index in range(3)] == ["10", "11", "12"]
    assert navigator.selector.currentData() == 12 and seen == [3]
    navigator.step(1)  # at the end: nothing
    assert navigator.selector.currentIndex() == 2 and seen == [3]
    navigator.select(11)
    assert navigator.selector.currentIndex() == 1 and seen == [3]


def test_n2_ints_footer_and_keyed_panels_mount_the_shared_navigator(qapp):
    from xdart.gui.tabs.scattering.scientific_view import ScientificView, _KeyedImagePanel
    from xdart.gui.tabs.scattering.shell_values import PanelKey, PanelRole
    from xdart.gui.tabs.scattering.shell_widgets import FrameNavigator, FrameSteps
    view = ScientificView()
    # the footer keeps its three items in its own order (its frozen layout rows): it places the shared steps' widgets
    assert isinstance(view.frame_steps, FrameSteps)
    assert view.frame_steps.widgets == (view.previous_frame, view.frame_selector, view.next_frame)
    assert all(view.footer.indexOf(widget) >= 0 for widget in view.frame_steps.widgets)
    assert view.previous_frame.toolTip() == "Show the previous frame."
    panel = _KeyedImagePanel(view, PanelKey(PanelRole.SLICE_2D, "probe"))
    assert isinstance(panel.navigator, FrameNavigator)
    assert panel.navigator.widgets == (panel.previous, panel.selector, panel.next)
    assert panel.previous.toolTip() == "Show the previous frame or slice, keeping the selected regions."
    # one step per click: the navigator steps the footer's selector, nothing else does
    keys = [object() for _ in range(3)]
    for index, key in enumerate(keys):
        view.frame_selector.add_frame(f"f{index}", key, f"scan:{index}")
    view.frame_selector.setCurrentIndex(0)
    seen = []
    view.frame_selector.currentIndexChanged.connect(seen.append)
    view.next_frame.setEnabled(True)
    view.next_frame.click()
    assert seen == [1]


def test_n3_the_source_cards_frame_row_mounts_the_navigator(qapp, tmp_path):
    from xdart.gui.analysis.background_template_row import FitSourceRow
    from xdart.gui.tabs.scattering.shell_widgets import FrameNavigator
    row = FitSourceRow()
    row.set_help_topic("fit.source")  # as both windows mount it
    frames = row.frames_row
    assert isinstance(frames.editor, FrameNavigator) and frames.label.text() == "Frame"
    assert frames.toolTip() and frames.help_button is not None
    layout = frames.layout()
    assert layout.itemAt(layout.count() - 1).widget() is frames.help_button  # its "?" last, in the "?" column
    row.show()
    assert frames.isVisible() and not row.frame_row.isVisible()  # Int's display: the navigator, no typed frame
    chosen = []
    row.frameChosen.connect(chosen.append)
    navigator = frames.editor
    source, keys = _selector(3, 1)
    row.mirror_frames(source)
    assert chosen == [] and navigator.selector.currentIndex() == 1
    navigator.next.click()
    assert chosen == [keys[2]]
    path = _result(tmp_path / "sample_int1d.nexus")
    row._file_chooser = lambda: str(path)
    row._ask_frame = lambda _path, _frame, _mode: (3, "default")
    row.choose_btn.click()
    row.set_frame_labels((0, 1, 2, 3, 4))
    assert navigator.selector.currentData() == 3 and frames.isVisible() and not row.frame_row.isVisible()
    changed = []
    row.changed.connect(lambda: changed.append(row.source.frame))
    navigator.previous.click()
    assert chosen[-1] == 2 and row.source.frame == 3  # the page answers a step
    row.step_frame(2)
    assert row.source.frame == 2 and changed == [2] and navigator.selector.currentData() == 2
    row.set_busy("A batch is fitting; step frames when it is done.")
    assert not navigator.isEnabled() and "batch is fitting" in frames.toolTip()
    row.set_busy("")
    assert navigator.isEnabled() and "batch is fitting" not in frames.toolTip()
    xye = tmp_path / "curve.xye"
    np.savetxt(xye, np.column_stack((Q, _peak(2.2), np.ones(Q.size))))
    row._file_chooser = lambda: str(xye)
    row._ask_unit = lambda _path, _hint: "q_A^-1"
    row.choose_btn.click()
    assert row.source.kind == "xye" and not frames.isVisible()  # one curve: no frames to step


def test_n4_a_nexus_source_steps_its_own_frames_and_a_batch_disables_the_navigator(qapp, tmp_path):
    from tests.xdart.test_fit_source_gui import _choose_nexus, _drain, _window
    path = _result(tmp_path / "sample_int1d.nexus")
    page = _page()
    try:
        dialog, host = _window(page, "peak")
        navigator = dialog.source_row.frames_row.editor
        _choose_nexus(dialog, path, 3)
        _drain(qapp, host)
        assert [navigator.selector.itemText(index) for index in range(navigator.selector.count())] == [
            "0", "1", "2", "3", "4"]
        assert navigator.selector.currentData() == 3
        navigator.next.click()
        _drain(qapp, host)
        assert dialog.fit_source().frame == 4
        assert np.array_equal(dialog._x, _stored(Q)) and np.array_equal(dialog._y, _stored(_peak(2.4)))
        assert page._analysis_identity is None  # Int is not asked
        dialog.batch_btn.click()
        assert host.computing and not navigator.isEnabled()
        _drain(qapp, host)
        assert navigator.isEnabled() and len(host.outcomes) == 5
        assert len(dialog.trend.table()[0]) == 5
        # a step within the result keeps its batch: the trend and the rows Save CSV writes stay, the frame is read
        navigator.previous.click()
        _drain(qapp, host)
        assert dialog.fit_source().frame == 3 and np.array_equal(dialog._y, _stored(_peak(2.3)))
        assert len(dialog.trend.table()[0]) == 5 and len(host.outcomes) == 5
        assert host.outcomes[3].result is None  # the compact batch kept frame 3's row only
        dialog.fit_btn.click()  # a re-fit of the batch's frame replaces its row; the batch's others stay
        _drain(qapp, host)
        assert len(dialog.trend.table()[0]) == 5 and len(host.outcomes) == 5
        assert host.outcomes[3].result is not None and host.outcomes[3].ok
        assert dialog.trend.table()[1] == (0.0, 1.0, 2.0, 3.0, 4.0)
    finally:
        _close_page(page, qapp)


_EIGER = (Path(os.environ.get("XDART_TEST_DATA", "/nonexistent")) / "xdart_processed_data"
          / "eiger_S069Ta_redo_eta2p0_1_scan001_int1d.nexus")


@pytest.mark.skipif(not _EIGER.is_file(), reason="needs XDART_TEST_DATA (xdart_processed_data/eiger_..._int1d.nexus)")
def test_n5_following_ints_display_it_lists_ints_frames_and_a_step_moves_int_then_this_window(
        qapp, tmp_path, monkeypatch, _xdart_qt_harness):
    from tests.xdart.scattering.test_p34_reintegrate_operation import _wait
    from tests.xdart.scattering.test_p3_experiment_operation_composition import _page as _real_page
    from tests.xdart.test_fit_source_gui import _window
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus
    page, _store = _real_page(tmp_path, monkeypatch)
    try:
        browse = page._context_controller.begin_browse(str(_EIGER), source_root=str(_EIGER.parent))
        outcome = _wait(page._context_controller.poll_browse, timeout=60)
        assert outcome.request is browse and outcome.status is BrowseLoadStatus.READY, outcome
        page.resize(1200, 800)
        page.show()
        page._refresh_shell()

        def painted():
            page._drain_executor()
            value = page._last_scientific_projection
            return value if value is not None and value.traces else None
        _wait(painted, timeout=60)
        dialog, _host = _window(page, "peak")
        footer = page._shell.scientific.frame_selector
        navigator = dialog.source_row.frames_row.editor
        assert _until(qapp, lambda: navigator.selector.count() == footer.count() > 1)
        assert navigator.selector.currentIndex() == footer.currentIndex()
        start = footer.currentIndex()
        target = start + 1 if start + 1 < footer.count() else start - 1
        wanted = footer.itemData(target)
        navigator.selector.setCurrentIndex(target)  # the user's pick
        assert footer.currentIndex() == target  # Int's own selector moved

        def followed():
            page._drain_executor()
            page._refresh_shell()
            trace, _anchor = page._capture_analysis_trace("peak")
            return (trace is not None and page._context_controller.navigation.current is wanted
                    and dialog._y is not None and np.array_equal(dialog._y, np.asarray(trace.intensity, dtype=float)))
        assert _until(qapp, followed, 60), "the window shows the stepped frame's trace"
        assert navigator.selector.currentIndex() == target
    finally:
        _close_page(page, qapp)


def test_n6_phase_analysis_has_the_same_frame_row_and_a_batch_holds_it(qapp):
    """Both windows (his "Mirror Int's frames"): Phase Analysis's Source card mirrors Int's footer in its Frame row, and
    its Batch holds the row as the Peak Fitter's does."""
    from tests.xdart.test_fit_source_gui import _window
    page = _page()
    try:
        dialog, _host = _window(page, "phase")
        row = dialog.source_row
        assert row.frames_row.isVisible() and row.frames_row.help_button is not None
        footer = page._shell.scientific.frame_selector
        keys = [object() for _ in range(3)]
        for index, key in enumerate(keys):
            footer.add_frame(f"f{index}", key, f"scan:{index}")
        footer.setCurrentIndex(1)
        page._sync_fit_frames("phase")
        navigator = row.frames_row.editor
        assert navigator.selector.count() == 3 and navigator.selector.currentData() is keys[1]
        dialog.set_batch_running(True)
        assert not navigator.isEnabled() and "batch is fitting" in row.frames_row.toolTip()
        dialog.set_batch_running(False)
        assert navigator.isEnabled()
    finally:
        _close_page(page, qapp)
