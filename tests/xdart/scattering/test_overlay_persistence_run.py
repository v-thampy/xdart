"""Overlay keeps its curves through a Live Run's scans, its end-of-Run Browse and the next Run, on the real page.

canonical_grid_overlay.md: only Clear, an incompatible grid or a true reintegration resets an Overlay. Maintainer,
2026-10-26: "it shouldn't clear at all when in the same run or different runs"; "the whole idea of overlay is
persistence so scans and frames can be compared unless chosen otherwise"; "if the axis changes, say q to q_ip, then
it makes sense that they can't be plotted together. otherwise they should be overlaid".
"""
from __future__ import annotations

from dataclasses import replace
import os
import weakref

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
    AxisProjection, BrowseTraceSnapshot, FrameNavigationProjection, ScientificProjection, ShellCommand,
    ShellCommandKind, TraceProjection,
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
        if plot_mode == "Overlay":
            # Run and Browse name a scan's curves alike, by the scan, never by its output's slot (maintainer,
            # 2026-10-26: "int1d should never be part of the legend in a 1d plot. It is totally redundant").
            names = [item.opts.get("name") for item in page._shell.scientific.curve.listDataItems()]
            assert names == ["a_1", "a_2", "b_1", "b_2"], names
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


def test_setting_background_updates_every_retained_overlay_curve(qapp, tmp_path):
    """Codex's event-2 HOLD (LIVE-1218-VIEWER2D-BG): Set BG in Int 1D Overlay corrects every curve shown, the kept
    curve of the earlier scan too; Clear BG brings each back as it was."""
    intent = _directory(tmp_path, live=True, output_mode="Append", mode="Int 1D")
    with _page(qapp, intent) as (page, view, events):
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Overlay"))
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_OPTION, 0.0, ("overlay", "offset")))
        _live_run(qapp, page, events, tmp_path / "raw", [("a", 1), ("b", 1)], "int1d")
        assert _rows(page) == [("a_int1d.nexus", 1), ("b_int1d.nexus", 1)]
        first, second = (item.yData.copy() for item in view.curve.listDataItems())
        assert np.isfinite(second).any() and np.nanmax(second) > 0
        view.background.click()
        assert _pump(qapp, lambda: view.background.text() == "Clear 1D BG")
        assert _rows(page) == [("a_int1d.nexus", 1), ("b_int1d.nexus", 1)]
        corrected_first, corrected_second = (item.yData for item in view.curve.listDataItems())
        scale = 1e-6 * float(np.nanmax(np.abs(second)))  # the drawn copies are float32 (display precision)
        np.testing.assert_allclose(corrected_second, second - second, atol=scale)
        np.testing.assert_allclose(corrected_first, first - second, atol=scale)
        # Clear BG brings every curve back as it was, the kept one too (it used to wipe the kept curves)
        view.background.click()
        assert _pump(qapp, lambda: view.background.text() == "Set 1D BG")
        assert _pump(qapp, lambda: len(view.curve.listDataItems()) == 2)
        assert _rows(page) == [("a_int1d.nexus", 1), ("b_int1d.nexus", 1)]
        restored_first, restored_second = (item.yData for item in view.curve.listDataItems())
        np.testing.assert_allclose(restored_first, first, atol=scale)
        np.testing.assert_allclose(restored_second, second, atol=scale)


def test_a_background_set_after_the_browse_handoff_corrects_every_kept_curve(qapp, tmp_path):
    """A one-scan Live Run in Overlay (two frames, no background) ends into Browse; Set BG then corrects every curve
    with no note (the handoff's curves keep their native basis); after a second Run the kept curves and its own are
    each drawn as itself minus the background at its axis, which Clear BG shows (LIVE-1218-VIEWER2D-BG, his pick
    "Any scan or file")."""
    intent = _directory(tmp_path, live=True, output_mode="Append", mode="Int 1D")
    with _page(qapp, intent) as (page, view, events):
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Overlay"))
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_OPTION, 0.0, ("overlay", "offset")))
        _live_run(qapp, page, events, tmp_path / "raw", [("a", 1), ("a", 2)], "int1d")
        assert _rows(page) == [("a_int1d.nexus", 1), ("a_int1d.nexus", 2)]
        before = [item.yData.copy() for item in view.curve.listDataItems()]
        frames = tuple(page._context_controller.navigation.selected)
        # Set BG on one frame: a two-frame Browse selection after the handoff refuses Set BG ("Select at least one
        # display frame"), as in 1.2.17 (seen there too; outside this objective)
        page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_FRAME, frame=frames[-1], frames=frames[-1:]))
        assert _pump(qapp, lambda: len(page._context_controller.navigation.selected) == 1)
        view.background.click()
        assert _pump(qapp, lambda: view.background.text() == "Clear 1D BG"), page._notice_text
        page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_FRAME, frame=frames[-1], frames=frames))
        assert _pump(qapp, lambda: len(view.curve.listDataItems()) == 2)
        assert "so it is drawn as it is" not in view.background.toolTip()
        background = np.asarray(page._background_owner._result.values, dtype=np.float64)
        scale = 1e-6 * float(np.nanmax(np.abs(background)))
        for drawn, raw in zip((item.yData for item in view.curve.listDataItems()), before, strict=True):
            np.testing.assert_allclose(drawn, raw - background, atol=4 * scale)
        _live_run(qapp, page, events, tmp_path / "raw", [("b", 1)], "int1d")
        assert ("b_int1d.nexus", 1) in _rows(page) and ("a_int1d.nexus", 1) in _rows(page)
        assert page._background_owner.phase == "ACTIVE" and "so it is drawn as it is" not in view.background.toolTip()
        rows = _rows(page)
        bg_axis = np.asarray(page._background_owner._result.axes[0], dtype=np.float64)
        natives = [np.asarray(trace.background_basis[0], np.float64) for trace in view.trace_history_projections]
        corrected = [(axis, np.asarray(item.yData, np.float64))
                     for axis, item in zip(natives, view.curve.listDataItems(), strict=True)]
        assert len(corrected) == len(rows)
        view.background.click()  # Clear BG: the same curves as they are, the kept ones too
        assert _pump(qapp, lambda: view.background.text() == "Set 1D BG" and len(view.curve.listDataItems()) == len(rows))
        assert _rows(page) == rows
        for (x, got), item in zip(corrected, view.curve.listDataItems(), strict=True):
            raw = np.asarray(item.yData, np.float64)
            on_bins = x.shape == bg_axis.shape and np.allclose(x, bg_axis, rtol=1e-6)
            want = raw - (background if on_bins else np.interp(x, bg_axis, background, left=np.nan, right=np.nan))
            finite = np.isfinite(want)
            assert finite.any()
            np.testing.assert_allclose(got[finite], want[finite], atol=4 * scale + 1e-6 * float(np.nanmax(np.abs(raw))))


@pytest.mark.parametrize("mode", ("Average", "Sum"))
def test_average_and_sum_fold_the_curves_kept_from_earlier_scans(qapp, tmp_path, mode):
    """His pick, 2026-11-22 ("In 1.2.18" for "Average/Sum drops the Overlay curves kept from earlier scans"): an
    Overlay of two scans (unequal curves on one grid) switched to Average or Sum folds every curve, the earlier
    scan's kept curve too, and its name says so; back in Overlay all are drawn again. Before, the kept curve left
    the fold."""
    intent = _directory(tmp_path, live=True, output_mode="Append", mode="Int 1D")
    with _page(qapp, intent) as (page, view, events):
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Overlay"))
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_OPTION, 0.0, ("overlay", "offset")))
        _live_run(qapp, page, events, tmp_path / "raw", [("a", 1), ("b", 1), ("b", 2)], "int1d")
        shown = [("a_int1d.nexus", 1), ("b_int1d.nexus", 1), ("b_int1d.nexus", 2)]
        assert _rows(page) == shown
        curves = [np.asarray(item.yData, np.float64) for item in view.curve.listDataItems()]
        assert len(curves) == 3 and not np.allclose(curves[1], curves[2])
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, mode))
        assert _pump(qapp, lambda: len(view.curve.listDataItems()) == 1)
        (folded,) = view.curve.listDataItems()
        expected = np.mean(curves, axis=0) if mode == "Average" else np.sum(curves, axis=0)
        np.testing.assert_allclose(np.asarray(folded.yData, np.float64), expected, rtol=1e-5)
        assert "3 curves, 1 kept" in folded.name()
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Overlay"))
        assert _pump(qapp, lambda: len(view.curve.listDataItems()) == 3)
        assert _rows(page) == shown
        view.background.click()
        assert _pump(qapp, lambda: view.background.text() == "Clear 1D BG")
        background = np.mean(curves[1:], axis=0)
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, mode))
        (corrected,) = view.curve.listDataItems()
        np.testing.assert_allclose(corrected.yData, expected - background * (3 if mode == "Sum" else 1),
                                   atol=1e-5 * np.max(np.abs(expected)))
        assert "3 curves, 1 kept" in corrected.name()


@pytest.mark.parametrize("mode", ("Int 1D", "Int 2D"))
def test_set_bg_on_a_two_frame_browse_selection_after_a_live_handoff(qapp, tmp_path, mode):
    """His pick, 2026-11-22 ("In 1.2.18" for "Set BG refuses a two-frame Browse selection right after a Live Run"):
    a one-scan Live Overlay Run (frames 1 and 2, unequal) ends into Browse with both selected; Set BG takes their
    mean from the Browse display's detached patterns (frame 1 has no context payload there), and each is drawn
    minus it: equal and opposite. Before, it refused ("Select at least one display frame"). Int 2D's Set takes maps,
    which Browse holds for the current frame only: it still refuses a two-frame selection, saying to select one."""
    intent = _directory(tmp_path, live=True, output_mode="Append", mode=mode)
    with _page(qapp, intent) as (page, view, events):
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Overlay"))
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_OPTION, 0.0, ("overlay", "offset")))
        _live_run(qapp, page, events, tmp_path / "raw", [("a", 1), ("a", 2)], "int1d" if mode == "Int 1D" else "int2d")
        assert len(page._context_controller.navigation.selected) == 2
        if mode == "Int 2D":
            view.background.click()
            assert _pump(qapp, lambda: "in Browse, select one frame" in page._notice_text), page._notice_text
            assert page._background_owner.phase in {"EMPTY", "RELEASED"}
            return
        first, second = (np.asarray(item.yData, np.float64) for item in view.curve.listDataItems())
        view.background.click()
        assert _pump(qapp, lambda: view.background.text() == "Clear 1D BG"), page._notice_text
        assert _pump(qapp, lambda: len(view.curve.listDataItems()) == 2)
        mean = (first + second) / 2
        corrected = [np.asarray(item.yData, np.float64) for item in view.curve.listDataItems()]
        scale = 1e-5 * float(np.nanmax(np.abs(mean)))
        np.testing.assert_allclose(corrected[0], first - mean, atol=scale)
        np.testing.assert_allclose(corrected[1], second - mean, atol=scale)
        assert np.nanmax(np.abs(corrected[0])) > 100 * scale


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


def test_browse_average_includes_a_frame_previously_excluded_by_overlay(qapp):
    path = "/out/scan_int1d.nexus"
    state, navigation = _gi_run_view(path, "scan", (1, 2))
    first, second = state.traces
    second = replace(second, axis=AxisProjection(second.axis.values + 0.25, "Q", "Å⁻¹"))
    state = replace(state, acquisition_selected=False, browse_artifact=path, traces=(first, second))
    frames = navigation.frames

    def snapshot(mode):
        projection = replace(state, plot_mode=mode)
        receipt = BrowseTraceSnapshot(frames, frames, (1, 2), plot_mode=mode,
                                      stacked_options_applied=mode == "Overlay",
                                      science_contract=projection.browse_science_contract)
        return replace(projection, browse_trace_snapshot=receipt)

    view = ScientificView()
    try:
        view.reconcile(snapshot("Overlay"), navigation, completed=2, total=2, detail="")
        assert view.excluded_trace_keys == (first.frame,)
        view.reconcile(snapshot("Average"), navigation, completed=2, total=2, detail="")
        assert view.excluded_trace_keys == ()
        assert view.trace_history_keys == frames
        (item,) = view.curve.listDataItems()
        axis, values = item.getData()
        projected = [np.interp(axis, trace.axis.values, trace.intensity, left=np.nan, right=np.nan)
                     for trace in (first, second)]
        np.testing.assert_allclose(values, np.nanmean(projected, axis=0))
    finally:
        view.close()


def test_an_incompatible_kept_axis_is_not_labelled_as_a_successful_fold(qapp):
    first, first_navigation = _gi_run_view("/out/first.nxs", "first", (1,))
    second, second_navigation = _gi_run_view("/out/second.nxs", "second", (2,))
    current, = second.traces
    current = replace(current, axis=AxisProjection(current.axis.values, "2θ", "°"))
    view = ScientificView()
    try:
        view.reconcile(first, first_navigation, completed=1, total=1, detail="")
        view.reconcile(replace(second, plot_mode="Sum", traces=(current,)), second_navigation,
                       completed=2, total=2, detail="")
        (item,) = view.curve.listDataItems()
        np.testing.assert_array_equal(item.yData, first.traces[0].intensity)
        assert item.name() == first.traces[0].title
    finally:
        view.close()


def test_clearing_the_scientific_view_releases_the_named_fold(qapp):
    first, first_navigation = _gi_run_view("/out/first.nxs", "first", (1,))
    second, second_navigation = _gi_run_view("/out/second.nxs", "second", (2,))
    view = ScientificView()
    try:
        view.reconcile(first, first_navigation, completed=1, total=1, detail="")
        view.reconcile(replace(second, plot_mode="Average"), second_navigation,
                       completed=2, total=2, detail="")
        assert view.curve.listDataItems()[0].name() == "Average · 2 curves, 1 kept"
        intensity = weakref.ref(view._named_aggregate[2].intensity)
        assert view.clear_workspace()
        assert intensity() is None
    finally:
        view.close()


def test_waterfall_options_do_not_remove_kept_rows_from_average(qapp):
    first, first_navigation = _gi_run_view("/out/first.nxs", "first", (1,))
    second, second_navigation = _gi_run_view("/out/second.nxs", "second", (2,))
    average = replace(second, plot_mode="Average")
    expected = (first.traces[0].intensity + second.traces[0].intensity) / 2
    view = ScientificView()
    try:
        view.reconcile(first, first_navigation, completed=1, total=1, detail="")
        view.reconcile(average, second_navigation, completed=2, total=2, detail="")
        np.testing.assert_allclose(view.curve.listDataItems()[0].yData, expected)
        filtered = replace(average, plot_options=replace(average.plot_options, waterfall_start=2, waterfall_step=2))
        view.reconcile(filtered, second_navigation, completed=2, total=2, detail="")
        np.testing.assert_allclose(view.curve.listDataItems()[0].yData, expected)
        assert view.trace_history_keys == (*first_navigation.frames, *second_navigation.frames)
    finally:
        view.close()


def test_the_detached_budget_keeps_all_currently_selected_aggregate_rows(qapp, monkeypatch):
    from xdart.gui.tabs.scattering import browse_1d_display

    kept, kept_navigation = _gi_run_view("/out/kept.nxs", "kept", (1,))
    first, first_navigation = _gi_run_view("/out/first.nxs", "first", (2,))
    second, second_navigation = _gi_run_view("/out/second.nxs", "second", (3,))
    frames = (*first_navigation.frames, *second_navigation.frames)
    selected = FrameNavigationProjection(frames, frames[-1], frames)
    aggregate = replace(first, plot_mode="Sum", traces=(*first.traces, *second.traces))
    monkeypatch.setattr(browse_1d_display, "MAX_BROWSE_1D_DETACHED_BYTES", 512)
    view = ScientificView()
    try:
        view.reconcile(kept, kept_navigation, completed=1, total=1, detail="")
        view.reconcile(aggregate, selected, completed=3, total=3, detail="")
        assert view.trace_history_keys == frames
        (item,) = view.curve.listDataItems()
        np.testing.assert_allclose(item.yData, first.traces[0].intensity + second.traces[0].intensity)
        assert "detached display budget" in view.browse_comparison_notice
    finally:
        view.close()


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


def _divided(page, suffix):
    """Each drawn curve against its own saved frame divided by that frame's own I0 (its .txt)."""
    from pathlib import Path

    from xrd_tools.io import FrameViewReader

    view = page._shell.scientific
    items = view.curve.listDataItems()
    assert len(items) == len(view.trace_history_keys)
    for item, frame in zip(items, view.trace_history_keys, strict=True):
        with FrameViewReader(Path(str(frame.artifact))) as reader:
            saved = reader.read(frame.local_frame_label)
        i0 = {key.lower(): value for key, value in saved.metadata_numeric.items()}["i0"]
        np.testing.assert_allclose(item.yData, np.asarray(saved.intensity_1d) / i0, rtol=1e-5)


def test_overlay_with_a_norm_channel_keeps_every_scan_each_divided_by_its_own_i0(qapp, tmp_path):
    """Codex N-f3f77127: a monitor-selected Overlay through scan b, the end-of-Run Browse and the next Run keeps every
    curve, each divided by its own frame's I0, the channel never switching off while the selection spans scans."""
    a1, a2, b1, b2, c1 = ((f"{scan}_int1d.nexus", index)
                          for scan, index in (("a", 1), ("a", 2), ("b", 1), ("b", 2), ("c", 1)))
    intent = _directory(tmp_path, live=True, output_mode="Append", mode="Int 1D")
    with _page(qapp, intent) as (page, _scientific, events):
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Overlay"))
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_OPTION, 0.0, ("overlay", "offset")))
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_NORM_CHANNEL, "i0"))
        shown = _notices(page)
        first = _live_run(qapp, page, events, tmp_path / "raw", [("a", 1), ("a", 2), ("b", 1), ("b", 2)], "int1d")
        assert first == [[a1], [a1, a2], [a1, a2, b1], [a1, a2, b1, b2]]
        assert page._last_scientific_projection.norm_channel == "i0"
        _divided(page, "int1d")
        _live_run(qapp, page, events, tmp_path / "raw", [("c", 1)], "int1d")
        assert sorted(_rows(page)) == [a1, a2, b1, b2, c1]
        assert page._last_scientific_projection.norm_channel == "i0"
        _divided(page, "int1d")
        assert not [text for text in shown if "new overlay" in text.lower()]
