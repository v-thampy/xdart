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

import xdart.gui.tabs.static_scan.frame_projection_adapter as fpa_mod
from xdart.modules.frame_publication import publication_from_frame_view
from xrd_tools.core import FrameView, IntegrationResult1D


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
