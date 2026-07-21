"""X1 GUI-adoption Slice 1 corrections — X1-GUI-R1 / R2 / R3.

Production-wired against the real ``staticWidget.displayframe`` render and the
real adapter (the ``project_frame`` seam is the real headless function).

- R1: the pinned projection describes the SAME frame as raw/cake/title — the
  canonical current-display label (latest selected, browse anchor first), not
  ``frame_ids[0]``.
- R2: projection-store ownership is scan-qualified — the active-run store serves
  only its own scan; a paused browse of another scan uses the publication-backed
  source; neither match fails closed.
- R3: the projection memo identity includes canonical ``mode_1d``/``mode_2d`` — a
  mode change at one generation performs a fresh lookup.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtWidgets

from xdart.gui.tabs.static_scan.frame_projection_adapter import (
    FrameProjectionAdapter,
    ProjectionRequest,
)
from xdart.modules.frame_publication import (
    PublicationStore,
    publication_from_frame_view,
)
from xrd_tools.core import (
    FrameRecord,
    FrameView,
    IntegrationResult1D,
    IntegrationResult2D,
)
from xrd_tools.session import CapabilityState, FrameRecordStore


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _r1d(scale=1.0):
    radial = np.linspace(0.5, 3.5, 4)
    intensity = scale * np.array([2.0, 4.0, 8.0, 16.0])
    return IntegrationResult1D(
        radial=radial, intensity=intensity, sigma=np.sqrt(intensity), unit="q_A^-1")


def _view(label, *, meta=None):
    return FrameView.from_results(
        label=label, result_1d=_r1d(),
        metadata_raw=dict(meta or {"i0": 42.0}),
        source_path="/data/loaded.nxs", source_frame_index=label)


def _make_widget(monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    return staticWidget()


def _publish(display, *labels):
    for label in labels:
        display.publication_store.upsert(publication_from_frame_view(_view(label)))


def _qualified_store(scan_key, label, *, meta):
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(FrameRecord.from_view(_view(label, meta=meta)))
    store._xdart_scan_key = scan_key           # active run declares its scan
    return store


def _const(value):
    return lambda: value


# --------------------------------------------------------------------------- #
# X1-GUI-R1: pin the current frame, not the first selected frame
# --------------------------------------------------------------------------- #

def test_r1_multi_selection_pins_latest_not_first(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        _publish(display, 0, 1, 2)
        display.frame_ids = (0, 1, 2)          # first=0, latest=2
        display.update()
        proj = display._current_frame_projection
        assert proj is not None
        # Fail-before (frame_ids[0]) pinned 0; the current frame is the latest, 2.
        assert proj.label == 2
    finally:
        widget.close()
        widget.deleteLater()


def test_r1_one_shot_browse_anchor_takes_precedence(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        _publish(display, 0, 1, 2)
        display.frame_ids = (0, 1, 2)          # first=0, latest=2
        display._browse_one_shot_anchor_label = 1   # anchor wins over both
        display.update()
        proj = display._current_frame_projection
        assert proj is not None
        assert proj.label == 1                 # the valid anchor, not 0 and not 2
    finally:
        widget.close()
        widget.deleteLater()


def test_r1_terminal_rapid_selection_generation_wins(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        _publish(display, 0, 1, 2, 3)
        # Rapid selection: (0,1) then (2,3) — only the terminal selection's
        # current (latest) frame may be pinned.
        display.frame_ids = (0, 1)
        display.update()
        display.frame_ids = (2, 3)
        display.update()
        proj = display._current_frame_projection
        assert proj is not None
        assert proj.label == 3                 # terminal latest, never a stale 0
    finally:
        widget.close()
        widget.deleteLater()


# --------------------------------------------------------------------------- #
# X1-GUI-R2: scan-qualified projection-store ownership
# --------------------------------------------------------------------------- #

def test_r2_active_a_paused_browse_b_resume_a_reused_label():
    # Active run A owns frame 0; a paused browse loads scan B's frame 0 (reused
    # label) into the publication store.  The projection must follow the
    # requested scan, never the attached active store unconditionally.
    active_a = _qualified_store("A", 0, meta={"scan": "A"})
    browse_b = PublicationStore()
    browse_b.upsert(publication_from_frame_view(_view(0, meta={"scan": "B"})))
    adapter = FrameProjectionAdapter(_const(active_a), _const(browse_b))

    following_a = adapter.project(ProjectionRequest("A", 0, generation=0))
    assert following_a.metadata.raw["scan"] == "A"       # active store serves A

    paused_browse_b = adapter.project(ProjectionRequest("B", 0, generation=1))
    assert paused_browse_b.metadata.raw["scan"] == "B"   # browse B, NOT active A/0

    resumed_a = adapter.project(ProjectionRequest("A", 0, generation=2))
    assert resumed_a.metadata.raw["scan"] == "A"         # back to the active store


def test_r2_no_cross_scan_fallback_fails_closed():
    # Active run A owns frame 0; browsing scan C whose frame 0 is not loaded must
    # fail closed (absent), never fall back to A/0.
    active_a = _qualified_store("A", 0, meta={"scan": "A"})
    empty_browse = PublicationStore()                    # C not loaded
    adapter = FrameProjectionAdapter(_const(active_a), _const(empty_browse))

    projected = adapter.project(ProjectionRequest("C", 0, generation=0))
    assert projected is not None
    assert projected.present is False                    # no A/0 cross-scan leak
    assert "scan" not in projected.metadata.raw


def test_r2_matching_active_path_still_prefers_record_store():
    # The matching active path is unchanged: A's store wins over a stale
    # publication for the same label when the request is for A.
    active_a = _qualified_store("A", 0, meta={"scan": "A-store"})
    stale_pub = PublicationStore()
    stale_pub.upsert(publication_from_frame_view(_view(0, meta={"scan": "pub"})))
    adapter = FrameProjectionAdapter(_const(active_a), _const(stale_pub))

    projected = adapter.project(ProjectionRequest("A", 0, generation=0))
    assert projected.metadata.raw["scan"] == "A-store"


# --------------------------------------------------------------------------- #
# X1-GUI-R3: canonical modes in the projection/memo identity
# --------------------------------------------------------------------------- #

def _r2d():
    radial = np.linspace(-1.0, 1.0, 3)
    azimuthal = np.linspace(-0.5, 0.5, 2)
    intensity = np.arange(6.0).reshape(3, 2)
    return IntegrationResult2D(
        radial=radial, azimuthal=azimuthal, intensity=intensity,
        sigma=np.sqrt(intensity + 1.0),
        unit="qip_A^-1", azimuthal_unit="qoop_A^-1")


def _gi_2d_store(label):
    view = FrameView.from_results(
        label=label, result_2d=_r2d(), metadata_raw={"i0": 1.0},
        source_path="/data/gi.nxs", source_frame_index=label)
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(FrameRecord.from_view(view))
    return store


def test_r3_mode_change_same_generation_relooks_up_with_correct_capability():
    # The record's 2D view is present for its own mode ("default") and absent for
    # any other requested mode — so the two modes give distinct capabilities.
    store = _gi_2d_store(0)
    adapter = FrameProjectionAdapter(_const(store), _const(None))

    present = adapter.project(
        ProjectionRequest("s", 0, generation=0, mode_2d="default"))
    absent = adapter.project(
        ProjectionRequest("s", 0, generation=0, mode_2d="q_chi"))

    # Fail-before (mode-insensitive memo key) returned the FIRST projection for
    # the second request: one lookup, and the wrong (AVAILABLE) 2D capability for
    # the absent mode.
    assert adapter.lookup_count == 2
    assert absent is not present
    assert present.capabilities.integrated_2d.state is CapabilityState.AVAILABLE
    assert absent.capabilities.integrated_2d.state is not CapabilityState.AVAILABLE


def test_r3_same_identity_including_modes_reuses_one_pin():
    # The one-projection-per-generation behavior is preserved when the COMPLETE
    # identity (scan/frame/generation/purpose AND modes) is unchanged.
    store = _gi_2d_store(0)
    adapter = FrameProjectionAdapter(_const(store), _const(None))
    first = adapter.project(
        ProjectionRequest("s", 0, generation=0, mode_2d="default"))
    again = adapter.project(
        ProjectionRequest("s", 0, generation=0, mode_2d="default"))
    assert again is first
    assert adapter.lookup_count == 1
