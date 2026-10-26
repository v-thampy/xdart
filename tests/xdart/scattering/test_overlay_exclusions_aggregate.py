"""An Overlay's exclusions never remove selected science from Sum or Average, and its admission holds no array its
kept curves do not (Codex review N-85413c4d OV-R4 and OV-R5, its discriminators): the runtime projects an Overlay's
left-out rows again on an aggregate mode's first refresh; the verdict memo and grid keep only kept curves' values."""
from dataclasses import replace

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering.test_e6_norm_consumers import (
    _acquisition_parts, _add_artifact, _retain, _project, _build, _prefs,
)
from xdart.gui.tabs.scattering.scientific_view import ScientificView, _detached_root_bytes


@pytest.fixture
def qapp(_xdart_qt_harness):
    return QtWidgets.QApplication.instance()


@pytest.mark.parametrize("mode", ("Sum", "Average"))
def test_aggregate_mode_reprojects_overlay_exclusions_on_its_first_refresh(qapp, monkeypatch, mode):
    from xdart.gui.tabs.scattering import browse_1d_display
    parts = _acquisition_parts([(1, 1., {})], partitions=2)
    owner_b = _add_artifact(parts.display, "/out/b.nxs", "run.b")
    last, delta = _retain(parts.display, owner_b, "run.b", "/out/b.nxs", 2, 2., {})
    assert parts.runtime.accept_navigation(delta, plot_mode="Overlay")
    frames = parts.runtime.navigation.frames
    assert parts.runtime.select_navigation(last, frames)
    prefs = _prefs(plot_mode="Overlay")
    payloads = _project(parts, prefs)
    state = replace(_build(payloads, parts.runtime.navigation, prefs, parts.runtime.norm_aggregate),
                    acquisition_selected=True)
    assert len(state.traces) == 2
    expected = sum(trace.intensity for trace in state.traces)
    if mode == "Average":
        expected = expected / 2.
    limit = _detached_root_bytes((state.traces[-1],)) * 3 // 2
    assert _detached_root_bytes(state.traces) > limit
    monkeypatch.setattr(browse_1d_display, "MAX_BROWSE_1D_DETACHED_BYTES", limit)
    view = ScientificView()
    try:
        view.reconcile(state, parts.runtime.navigation, completed=2, total=2, detail="")
        assert view.trace_history_keys == (last,)
        assert parts.runtime.commit_navigation_projection(view.trace_history_keys, excluded=view.excluded_trace_keys)
        prefs = replace(prefs, plot_mode=mode)
        payloads = _project(parts, prefs)
        state = replace(_build(payloads, parts.runtime.navigation, prefs, parts.runtime.norm_aggregate),
                        acquisition_selected=True)
        view.reconcile(state, parts.runtime.navigation, completed=2, total=2, detail="")
        [curve] = view.curve.listDataItems()
        np.testing.assert_allclose(curve.yData, expected, err_msg="all selected rows belong in the aggregate")
    finally:
        view.close()


def test_admission_memo_does_not_retain_evicted_axis_roots(qapp, monkeypatch):
    from tests.xdart.scattering.test_overlay_admission_contract import _trace, _state
    from xdart.gui.tabs.scattering import browse_1d_display
    from xdart.gui.tabs.scattering.events import RunIdentity
    identity = RunIdentity(48, "review-memo-budget")
    older = _trace(identity, "a", "/out/a.nxs", 1)
    shown = _trace(identity, "b", "/out/b.nxs", 1)
    limit = _detached_root_bytes((shown,)) + 1
    monkeypatch.setattr(browse_1d_display, "MAX_BROWSE_1D_DETACHED_BYTES", limit)
    view = ScientificView()
    try:
        view.reconcile(*_state((older, shown)), completed=2, total=2, detail="")
        assert view.trace_history_keys == (shown.frame,)
        arrays = []
        for trace in view._trace_history_by_identity.values():
            arrays.extend((trace.axis.values, trace.intensity))
        arrays.extend(values for values, _verdict in view._comparison_verdicts.values())
        if view._comparison_grid is not None:
            arrays.append(view._comparison_grid[1])
        roots = {}
        for array in arrays:
            while isinstance(array.base, np.ndarray):
                array = array.base
            roots[id(array)] = array
        assert sum(array.nbytes for array in roots.values()) <= limit
    finally:
        view.close()
