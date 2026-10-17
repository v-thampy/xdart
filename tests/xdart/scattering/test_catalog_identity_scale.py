"""Catalogue identity checks at Show All scale (GUI-PERF-01).

Each row drives a real view through its production interface with a
5,000-frame catalogue, the size of a large live or Browse selection.  The rows
pin exact object identity, catalogue order, the bounded display extent and the
refusal of an equal-valued key that is not the catalogue's own object.

The work bound counts Python loop iterations (backward jumps) on the GUI
thread during one synchronous call, so it does not depend on machine speed or
load.  A nested identity scan costs about N(N+1)/2 = 12.5 million iterations
at this size; a linear check costs a few per frame.  Wall time is recorded as
a JUnit property, not asserted.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import replace

import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_values import (
    BrowseTraceSnapshot,
    FrameNavigationProjection,
    FrameSelectionIntent,
    ShellCommandKind,
)
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell

from tests.xdart.scattering.e3_shell_support import make_shell_projection

FRAME_COUNT = 5_000
LOOP_STEP_BOUND = 100 * FRAME_COUNT


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _loop_steps(interaction, record_property) -> int:
    """Run one synchronous call; return the GUI thread's loop iterations."""

    monitoring = sys.monitoring
    tool = next(tool for tool in (3, 4) if monitoring.get_tool(tool) is None)
    thread = threading.get_ident()
    steps = 0

    def jumped(_code, source, destination):
        nonlocal steps
        if destination < source and threading.get_ident() == thread:
            steps += 1

    monitoring.use_tool_id(tool, "catalogue identity scale")
    monitoring.register_callback(tool, monitoring.events.JUMP, jumped)
    monitoring.set_events(tool, monitoring.events.JUMP)
    started = time.perf_counter()
    try:
        interaction()
    finally:
        elapsed = time.perf_counter() - started
        monitoring.set_events(tool, monitoring.events.NO_EVENTS)
        monitoring.register_callback(tool, monitoring.events.JUMP, None)
        monitoring.free_tool_id(tool)
    record_property("loop_steps", steps)
    record_property("elapsed_s", round(elapsed, 4))
    return steps


def _catalogue():
    projection = make_shell_projection(
        frame_count=FRAME_COUNT,
        heavy_indices=(),
        plot_mode="Overlay",
        source_scan="scan",
    )
    return projection, projection.navigation.frames


def _same(left, right) -> bool:
    return len(left) == len(right) and all(
        first is second for first, second in zip(left, right)
    )


def _show_all(projection, logical, drawn):
    """Show All as Browse plans it: every frame logical, a bounded few drawn."""

    base = replace(
        projection.scientific,
        heavy=None,
        heavy_available=frozenset(),
    )
    state = replace(base, traces=tuple(base.traces[index] for index in drawn))
    return replace(
        state,
        browse_trace_snapshot=BrowseTraceSnapshot(
            logical,
            tuple(logical[index] for index in drawn),
            tuple(index + 1 for index in drawn),
            plot_mode="Overlay",
            stacked_options_applied=True,
            science_contract=state.browse_science_contract,
        ),
    )


def _reconcile(view, state, navigation) -> None:
    view.reconcile(
        state,
        navigation,
        completed=FRAME_COUNT,
        total=FRAME_COUNT,
        detail="Ready",
    )


def test_show_all_snapshot_adopts_the_exact_logical_extent_linearly(
    qapp: QtWidgets.QApplication,
    record_property,
) -> None:
    projection, frames = _catalogue()
    navigation = FrameNavigationProjection(frames, frames[0], frames)
    drawn = (0, 1250, 2500, 3750)
    view = ScientificView()
    try:
        _reconcile(view, _show_all(projection, frames, drawn), navigation)
        # A later receipt for the same selection is proven again in full.
        state = _show_all(projection, frames, drawn)
        steps = _loop_steps(
            lambda: _reconcile(view, state, navigation),
            record_property,
        )
        assert view.trace_row_count == FRAME_COUNT
        assert _same(view.trace_history_keys, frames)
        assert len(view.curve.listDataItems()) == len(drawn)
        assert steps <= LOOP_STEP_BOUND

        drawn = tuple(range(0, FRAME_COUNT, 20))
        _reconcile(view, _show_all(projection, frames, drawn), navigation)
        assert _same(view.trace_history_keys, frames)
        assert len(view.curve.listDataItems()) == len(drawn) <= 256
    finally:
        view.close()


def test_show_all_snapshot_refuses_an_equal_valued_foreign_frame_linearly(
    qapp: QtWidgets.QApplication,
    record_property,
) -> None:
    projection, frames = _catalogue()
    navigation = FrameNavigationProjection(frames, frames[0], frames)
    foreign = replace(frames[-1])
    assert foreign == frames[-1] and foreign is not frames[-1]
    drawn = (0, 1250, 2500, 3750)
    view = ScientificView()
    try:
        _reconcile(view, _show_all(projection, frames, drawn), navigation)
        state = _show_all(projection, (*frames[:-1], foreign), drawn)

        def refuse() -> None:
            with pytest.raises(ValueError, match="changed presentation scope"):
                _reconcile(view, state, navigation)

        assert _loop_steps(refuse, record_property) <= LOOP_STEP_BOUND
    finally:
        view.close()


def _browse_shell(qapp, state):
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    shell.resize(1200, 800)
    shell.show()
    shell.apply_state(state)
    qapp.processEvents()
    commands.clear()
    return shell, commands


def _without_traces(projection, navigation):
    return replace(
        projection,
        scientific=replace(projection.scientific, traces=()),
        navigation=navigation,
    )


def _select_all(shell) -> None:
    shell.browser.frames.setFocus()
    QtTest.QTest.keyClick(
        shell.browser.frames,
        QtCore.Qt.Key.Key_A,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    )
    assert shell.browser.frame_selection_pending


def test_pending_select_all_survives_a_reconcile_and_live_append_linearly(
    qapp: QtWidgets.QApplication,
    record_property,
) -> None:
    projection, frames = _catalogue()
    state = _without_traces(
        projection,
        FrameNavigationProjection(frames, frames[0], (frames[0],)),
    )
    shell, commands = _browse_shell(qapp, state)
    try:
        _select_all(shell)
        # Another projection lands inside the selection's quiet window.
        steps = _loop_steps(
            lambda: shell.apply_state(replace(state, revision=state.revision + 1)),
            record_property,
        )
        assert shell.browser.frame_selection_pending

        appended = DisplayFrameKey(
            frames[0].run_identity, "scan", "result.nxs", 1, FRAME_COUNT + 1,
        )
        catalogue = (*frames, appended)
        shell.apply_state(replace(
            state,
            revision=state.revision + 2,
            browser=replace(state.browser, frames=catalogue),
            navigation=FrameNavigationProjection(
                catalogue, frames[0], (frames[0],),
            ),
        ))
        assert shell.browser.frame_selection_pending
        QtTest.QTest.qWait(250)
        assert len(commands) == 1
        command = commands[0]
        assert command.kind is ShellCommandKind.SELECT_BROWSER_FRAMES
        assert command.intent is FrameSelectionIntent.EXACT
        assert command.frame is frames[0]
        assert _same(command.frames, frames)
        assert steps <= LOOP_STEP_BOUND
    finally:
        shell.close()


def test_pending_select_all_cancels_when_one_key_is_an_equal_valued_copy(
    qapp: QtWidgets.QApplication,
) -> None:
    projection, frames = _catalogue()
    state = _without_traces(
        projection,
        FrameNavigationProjection(frames, frames[0], (frames[0],)),
    )
    shell, commands = _browse_shell(qapp, state)
    try:
        _select_all(shell)
        foreign = replace(frames[-1])
        assert foreign == frames[-1] and foreign is not frames[-1]
        catalogue = (*frames[:-1], foreign)
        shell.apply_state(replace(
            state,
            revision=state.revision + 1,
            browser=replace(state.browser, frames=catalogue),
            navigation=FrameNavigationProjection(
                catalogue, frames[0], (frames[0],),
            ),
        ))
        assert not shell.browser.frame_selection_pending
        QtTest.QTest.qWait(250)
        assert commands == []
    finally:
        shell.close()


def test_overlay_revisit_of_the_highlighted_row_keeps_catalogue_order_linearly(
    qapp: QtWidgets.QApplication,
    record_property,
) -> None:
    projection, frames = _catalogue()
    # Accumulated traces, held in an order other than the catalogue's.
    accumulated = frames[1:][::-1]
    state = _without_traces(
        projection,
        FrameNavigationProjection(frames, frames[0], accumulated),
    )
    shell, commands = _browse_shell(qapp, state)
    try:
        browser = shell.browser
        # The page moves current (the plot footer, say); the highlight stays.
        shell.apply_state(replace(
            state,
            revision=state.revision + 1,
            navigation=FrameNavigationProjection(frames, frames[2], accumulated),
        ))
        qapp.processEvents()
        commands.clear()
        assert tuple(
            index.row()
            for index in browser.frames.selectionModel().selectedRows()
        ) == (0,)
        assert browser.frames.currentIndex().row() == 2
        assert _same(browser._trace_frames, accumulated)

        index = browser.frame_model.index(0, 0)
        browser.frames.scrollTo(index)
        qapp.processEvents()
        steps = _loop_steps(
            lambda: QtTest.QTest.mouseClick(
                browser.frames.viewport(),
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
                browser.frames.visualRect(index).center(),
            ),
            record_property,
        )
        assert _same(browser._trace_frames, frames)
        QtTest.QTest.qWait(250)
        assert len(commands) == 1
        command = commands[0]
        assert command.kind is ShellCommandKind.SELECT_BROWSER_FRAMES
        assert command.intent is FrameSelectionIntent.VISIT
        assert command.frame is frames[0]
        assert _same(command.frames, (frames[0],))
        assert steps <= LOOP_STEP_BOUND
    finally:
        shell.close()
