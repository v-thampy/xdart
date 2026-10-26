"""Overlay admission on ScientificView's real projections (Codex review N-f3f77127, OV-R1 to OV-R3, its
discriminators): the curve on show survives an incompatible predecessor; Browse draws only the rows it admits; an
unchanged mismatch is told once; a Run's earlier scans obey the detached display budget."""
from dataclasses import replace
import logging

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xrd_tools.core import FrameView
from xrd_tools.core.frame_view import axis_from_unit
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey, StandardDisplayPayload
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.scientific_axes import trace_projection
from xdart.gui.tabs.scattering.scientific_view import ScientificView, _detached_root_bytes
from xdart.gui.tabs.scattering.shell_values import BrowseTraceSnapshot, FrameNavigationProjection, ScientificProjection


@pytest.fixture
def qapp(_xdart_qt_harness):
    return QtWidgets.QApplication.instance()


def _trace(identity, scan, artifact, label, *, wavelength=1e-10, axis="Q", scale=1.):
    frame = DisplayFrameKey(identity, scan, artifact, label, label)
    native = FrameView(label, axis_1d=axis_from_unit("q_A^-1", np.array([0.1, 0.2, 0.3]) * scale),
                       intensity_1d=np.array([1., 2., 3.]) * label)
    payload = StandardDisplayPayload(0, frame, scan, native, wavelength_m=wavelength)
    trace = trace_projection(payload, requested_axis=axis, allow_cake=False, slice_enabled=False,
                             slice_center=0., slice_width=1.)
    assert trace is not None
    return trace


def _state(traces, *, axis="Q"):
    frames = tuple(trace.frame for trace in traces)
    return (ScientificProjection(traces=traces, processing_mode="Int 1D", plot_mode="Overlay", plot_axis=axis,
                                 acquisition_selected=True), FrameNavigationProjection(frames, frames[-1], frames))


def test_energy_scan_converted_grid_keeps_its_current_curve(qapp):
    identity = RunIdentity(41, "review-overlay")
    first = _trace(identity, "energy", "/out/energy.nexus", 1, wavelength=7.75e-11, axis="2theta")
    last = _trace(identity, "energy", "/out/energy.nexus", 2, wavelength=7.29e-11, axis="2theta")
    assert not np.allclose(first.axis.values, last.axis.values, rtol=1e-5, atol=1e-8)
    state, navigation = _state((first, last), axis="2theta")
    view = ScientificView()
    try:
        view.reconcile(state, navigation, completed=2, total=2, detail="")
        assert view.trace_history_keys == (last.frame,), "incompatible old row may leave; current row must survive"
        assert len(view.curve.listDataItems()) == 1
    finally:
        view.close()


def test_browse_comparison_consumes_its_filtered_incoming_rows(qapp):
    identity = RunIdentity(42, "review-overlay")
    first = _trace(identity, "energy", "/out/energy.nexus", 1, wavelength=7.75e-11, axis="2theta")
    last = _trace(identity, "energy", "/out/energy.nexus", 2, wavelength=7.29e-11, axis="2theta")
    state, navigation = _state((first, last), axis="2theta")
    state = replace(state, acquisition_selected=False, browse_artifact="/out/energy.nexus")
    snapshot = BrowseTraceSnapshot(navigation.frames, navigation.frames, (1, 2), plot_mode="Overlay",
                                   stacked_options_applied=True, science_contract=state.browse_science_contract)
    state = replace(state, browse_trace_snapshot=snapshot)
    view = ScientificView()
    try:
        view.reconcile(state, navigation, completed=2, total=2, detail="")
        assert len(view.curve.listDataItems()) == 1, "a refused row must not be drawn by Browse"
        assert view.trace_history_keys == (last.frame,)
    finally:
        view.close()


def test_replayed_mismatch_emits_one_reset_notice(qapp, caplog):
    identity = RunIdentity(43, "review-overlay")
    first = _trace(identity, "a", "/out/a.nexus", 1)
    last = _trace(identity, "b", "/out/b.nexus", 1, scale=2.)
    state, navigation = _state((first, last))
    view = ScientificView()
    caplog.set_level(logging.INFO)
    try:
        for _ in range(3):
            view.reconcile(state, navigation, completed=2, total=2, detail="")
        assert view.trace_history_keys == (last.frame,)
        resets = [record.getMessage() for record in caplog.records if "Started a new overlay" in record.getMessage()]
        assert len(resets) == 1, resets
    finally:
        view.close()


def test_same_run_prior_scan_rows_obey_detached_budget(qapp, monkeypatch):
    from xdart.gui.tabs.scattering import browse_1d_display
    identity = RunIdentity(44, "review-overlay")
    first = _trace(identity, "a", "/out/a.nexus", 1)
    last = _trace(identity, "b", "/out/b.nexus", 1)
    limit = _detached_root_bytes((last,)) * 3 // 2
    assert _detached_root_bytes((first, last)) > limit
    monkeypatch.setattr(browse_1d_display, "MAX_BROWSE_1D_DETACHED_BYTES", limit)
    state, navigation = _state((first, last))
    view = ScientificView()
    try:
        view.reconcile(state, navigation, completed=2, total=2, detail="")
        kept = tuple(view._trace_history_by_identity.values())
        assert _detached_root_bytes(kept) <= limit
        assert view.trace_history_keys == (last.frame,)
    finally:
        view.close()


def _browse(traces, artifact):
    """The same curves as a Browse of ``artifact`` hands them over: a typed snapshot of that file's rows."""
    state, navigation = _state(traces)
    state = replace(state, acquisition_selected=False, browse_artifact=artifact)
    snapshot = BrowseTraceSnapshot(navigation.frames, navigation.frames, tuple(range(1, len(traces) + 1)),
                                   plot_mode="Overlay", stacked_options_applied=True,
                                   science_contract=state.browse_science_contract)
    return replace(state, browse_trace_snapshot=snapshot), navigation


@pytest.mark.parametrize("first_from", ["Run", "Browse"])
def test_older_history_leaves_whole_past_the_budget_between_run_and_browse(qapp, monkeypatch, first_from):
    """Run curves a Browse keeps, and Browse curves a Run keeps, are one detached history under one budget: past it
    the older file's curves leave, with the notice, and the shown file's stay."""
    from xdart.gui.tabs.scattering import browse_1d_display
    identity = RunIdentity(45, "review-overlay")
    older = _trace(identity, "a", "/out/a.nexus", 1)
    shown = _trace(identity, "b", "/out/b.nexus", 1)
    limit = _detached_root_bytes((shown,)) * 3 // 2
    assert _detached_root_bytes((older, shown)) > limit
    monkeypatch.setattr(browse_1d_display, "MAX_BROWSE_1D_DETACHED_BYTES", limit)
    view = ScientificView()
    try:
        first = _state((older,)) if first_from == "Run" else _browse((older,), "/out/a.nexus")
        view.reconcile(*first, completed=1, total=1, detail="")
        assert view.trace_history_keys == (older.frame,)
        second = _browse((shown,), "/out/b.nexus") if first_from == "Run" else _state((shown,))
        view.reconcile(*second, completed=1, total=1, detail="")
        assert view.trace_history_keys == (shown.frame,)
        assert _detached_root_bytes(tuple(view._trace_history_by_identity.values())) <= limit
        assert view.browse_comparison_notice == "Started a new overlay: detached display budget reached."
    finally:
        view.close()


def test_a_kept_curve_stays_until_the_runs_own_spelling_of_its_frame_is_drawn(qapp):
    """Codex N-85413c4d's persistence observation (XDART_OVERLAY_DEBUG: held b_int1d:2 left once the next Run selected
    its own b:2, though b:2 had no curve yet): a frame drawn once in Browse's spelling and selected again in a Run's
    stays drawn, once, until the Run's own curve for it arrives."""
    browse = RunIdentity(46, "review-overlay")
    run = RunIdentity(47, "review-overlay")
    b1, b2 = (_trace(browse, "b_int1d", "/out/b.nexus", label) for label in (1, 2))
    run_b2, run_c1 = _trace(run, "b", "/out/b.nexus", 2), _trace(run, "c", "/out/c.nexus", 1)
    view = ScientificView()
    try:
        view.reconcile(*_browse((b1, b2), "/out/b.nexus"), completed=2, total=2, detail="")
        assert view.trace_history_keys == (b1.frame, b2.frame)
        state, navigation = _state((run_b2, run_c1))
        view.reconcile(replace(state, traces=(run_c1,)), navigation, completed=2, total=2, detail="")
        assert view.trace_history_keys == (b1.frame, b2.frame, run_c1.frame)
        assert len(view.curve.listDataItems()) == 3
        view.reconcile(state, navigation, completed=2, total=2, detail="")
        assert view.trace_history_keys == (b1.frame, run_b2.frame, run_c1.frame)
        assert len(view.curve.listDataItems()) == 3
    finally:
        view.close()
