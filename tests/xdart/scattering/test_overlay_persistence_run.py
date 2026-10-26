"""Overlay keeps its curves through a Live Run's scans, its end-of-Run Browse and the next Run, on the real page.

canonical_grid_overlay.md: only Clear, an incompatible grid or a true reintegration resets an Overlay. Maintainer,
2026-10-26: "it shouldn't clear at all when in the same run or different runs"; "the whole idea of overlay is
persistence so scans and frames can be compared unless chosen otherwise"; "if the axis changes, say q to q_ip, then
it makes sense that they can't be plotted together. otherwise they should be overlaid".
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from tests.xdart.scattering.test_browse_reference_rings_run import _directory, _page, _pump, _TERMINAL
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.page import ScatteringWorkspaceShell
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_values import (
    AxisProjection, FrameNavigationProjection, ScientificProjection, ShellCommand, ShellCommandKind, TraceProjection,
)


@pytest.fixture
def qapp(_xdart_qt_harness):
    from pyqtgraph.Qt import QtWidgets
    return QtWidgets.QApplication.instance()


def _rows(page):
    """The curves drawn, each as its scan's output and frame label: a Run and a Browse of one scan name it alike."""
    return [(os.path.basename(str(frame.artifact)), frame.local_frame_label)
            for frame in page._shell.scientific.trace_history_keys]


def _settled(page, output):
    """The end-of-Run Browse of ``output`` has painted: its rows are Browse's, nothing is pending."""
    view = page._shell.scientific
    return (not page._context_controller.browse_pending and not page._scientific_repaint_pending
            and any(os.path.basename(str(frame.artifact)) == output for frame in view.trace_history_keys)
            and all(frame.source_scan == output.split(".")[0] for frame in view.trace_history_keys
                    if os.path.basename(str(frame.artifact)) == output))


def _live_run(qapp, page, events, raw, frames, suffix):
    """A Live Run: each (scan, index) frame is written once the one before is drawn; then Stop and its Browse."""
    shell = page.findChild(ScatteringWorkspaceShell)
    start = len(events)
    page._handle_shell_command(ShellCommand(ShellCommandKind.SET_LIVE, True))
    shell.run_controls.startButton.click()
    drawn = []
    for scan, index in frames:
        _frame(raw, scan, index, "txt", float(index))
        row = (f"{scan}_{suffix}.nexus", index)
        assert _pump(qapp, lambda: row in _rows(page), 30.), (row, _rows(page))
        drawn.append(_rows(page))
    shell.run_controls.stopButton.click()
    assert _pump(qapp, lambda: any(event.kind in _TERMINAL for event in events[start:]), 30.)
    last = f"{frames[-1][0]}_{suffix}.nexus"
    assert _pump(qapp, lambda: _settled(page, last), 30.), (_rows(page), page._notice_text)
    return drawn


def _notices(page):
    """Every notice the page shows from now on, in order."""
    shown, notice = [], page._notice
    page._notice = lambda text, *args, **kwargs: (shown.append(text), notice(text, *args, **kwargs))[1]
    return shown


def _curves(page, plot_mode):
    view = page._shell.scientific
    if plot_mode == "Waterfall":
        return len(view.waterfall_source_frame_keys)
    return len(view.curve.listDataItems())


@pytest.mark.parametrize(("mode", "plot_mode"), [("Int 1D", "Overlay"), ("Int 2D", "Overlay"),
                                                 ("Int 1D", "Waterfall")])
def test_overlay_keeps_every_scan_through_the_run_its_end_and_the_next_run(qapp, tmp_path, mode, plot_mode):
    suffix = "int1d" if mode == "Int 1D" else "int2d"
    a1, a2, b1, b2, c1 = ((f"{scan}_{suffix}.nexus", index)
                          for scan, index in (("a", 1), ("a", 2), ("b", 1), ("b", 2), ("c", 1)))
    intent = _directory(tmp_path, live=True, output_mode="Append", mode=mode)
    with _page(qapp, intent) as (page, _scientific, events):
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, plot_mode))
        shown = _notices(page)
        first = _live_run(qapp, page, events, tmp_path / "raw", [("a", 1), ("a", 2), ("b", 1), ("b", 2)], suffix)
        # Scan b's first frame keeps scan a's curves (the scan boundary cleared them before).
        assert first == [[a1], [a1, a2], [a1, a2, b1], [a1, a2, b1, b2]]
        # The end-of-Run Browse of scan b keeps scan a's curves beside its own, each once.
        assert _rows(page) == [a1, a2, b1, b2]
        assert _curves(page, plot_mode) == 4
        second = _live_run(qapp, page, events, tmp_path / "raw", [("c", 1)], suffix)
        # The next Run keeps them all, the Run's and Browse's spelling of one frame drawn once.
        assert sorted(second[-1]) == [a1, a2, b1, b2, c1]
        assert sorted(_rows(page)) == [a1, a2, b1, b2, c1]
        assert _curves(page, plot_mode) == 5
        assert not [text for text in shown if "new overlay" in text.lower()]


def test_overlay_starts_new_when_the_next_run_samples_another_grid(qapp, tmp_path):
    intent = _directory(tmp_path, live=True, output_mode="Append", mode="Int 1D")
    with _page(qapp, intent) as (page, _scientific, events):
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Overlay"))
        shown = _notices(page)
        _live_run(qapp, page, events, tmp_path / "raw", [("a", 1), ("b", 1)], "int1d")
        assert _rows(page) == [("a_int1d.nexus", 1), ("b_int1d.nexus", 1)]
        store = page._intents
        candidate = store.snapshot().thaw()
        candidate.bai_1d_args = {**candidate.bai_1d_args, "npt": 12}
        store.commit(candidate, expected_revision=store.revision)
        _live_run(qapp, page, events, tmp_path / "raw", [("c", 1)], "int1d")
        # Twelve points are not eight: the earlier curves leave, once, with the notice saying why.
        assert _rows(page) == [("c_int1d.nexus", 1)]
        assert _curves(page, "Overlay") == 1
        assert [text for text in shown if "new overlay" in text.lower()] == [
            "Started a new overlay: sampled axes do not match."]


def _gi_run_view(artifact, scan, labels):
    """A GI Run's view of one scan: its frames, all selected, a q curve each, as its payloads name them."""
    frames = tuple(DisplayFrameKey(RunIdentity(7, "overlay-persist"), scan, artifact, label, label) for label in labels)
    x = np.linspace(0.5, 4.0, 16)
    x.setflags(write=False)
    traces = []
    for label, frame in zip(labels, frames, strict=True):
        y = np.sin(x * label)
        y.setflags(write=False)
        traces.append(TraceProjection(frame, AxisProjection(x, "Q", "Å⁻¹"), y, f"{scan}:{label}"))
    state = ScientificProjection(traces=tuple(traces), processing_mode="Int 1D", plot_mode="Overlay",
                                 measurement_mode="GI", gi_mode_1d="q_total", acquisition_selected=True)
    return state, FrameNavigationProjection(frames, frames[-1], frames)


def test_a_gi_overlay_keeps_its_curves_through_a_new_runs_first_empty_state(qapp):
    """A state without curves has no payload to say it is GI, so a new Run's first state reads as Standard; it must
    not end the GI comparison it found."""
    first, first_navigation = _gi_run_view("/out/a_int1d.nexus", "a", (1, 2))
    second, second_navigation = _gi_run_view("/out/b_int1d.nexus", "b", (1,))
    empty = ScientificProjection(processing_mode="Int 1D", plot_mode="Overlay", acquisition_selected=True)
    assert (empty.measurement_mode, empty.gi_mode_1d) == ("Standard", "")
    view = ScientificView()
    try:
        view.reconcile(first, first_navigation, completed=0, total=0, detail="")
        assert view.trace_history_keys == first_navigation.frames
        view.reconcile(empty, FrameNavigationProjection(), completed=0, total=0, detail="")
        assert view.trace_history_keys == first_navigation.frames
        view.reconcile(second, second_navigation, completed=0, total=0, detail="")
        assert view.trace_history_keys == (*first_navigation.frames, *second_navigation.frames)
        assert len(view.curve.listDataItems()) == 3
    finally:
        view.close()
