"""X1 GUI-adoption Slice 1 — production-wired render pin (mutation-proof).

Drives the REAL ``staticWidget.displayframe`` render (``update()`` ->
``_update_impl`` -> ``_pin_selected_frame_projection``) against a real
``PublicationStore`` (browse path; no live record store).  Locks the two Slice-1
contracts on the actual production wiring:

* one ``project_frame`` lookup per selected-frame render generation (repeat
  renders at the same generation reuse the pinned value; a new selection
  generation performs exactly one more);
* removing the real ``_pin_selected_frame_projection()`` call in ``_update_impl``
  makes ``test_render_pins_one_projection_per_generation`` fail (the pinned
  ``_current_frame_projection`` goes ``None`` and the lookup count goes to 0).
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
from xrd_tools.session import FrameProjection


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
        metadata_raw=dict(meta or {"i0": 42.0, "temp": 300.0}),
        source_path="/data/loaded.nxs", source_frame_index=label)


def _spy_project_frame(monkeypatch):
    """Count real ``project_frame`` calls made through the adapter module."""
    calls = []
    real = fpa_mod.project_frame

    def _counted(store, label, **kwargs):
        calls.append(label)
        return real(store, label, **kwargs)

    monkeypatch.setattr(fpa_mod, "project_frame", _counted)
    return calls


def _make_widget(monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    return staticWidget()


def test_render_pins_one_projection_per_generation(qapp, monkeypatch, tmp_path):
    calls = _spy_project_frame(monkeypatch)
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        # Browse path: publish two frames into the shared store, no live run.
        display.publication_store.upsert(publication_from_frame_view(_view(0)))
        display.publication_store.upsert(publication_from_frame_view(_view(1)))

        # Select frame 0 and drive the REAL render.
        display.frame_ids = (0,)
        display.update()

        projection = display._current_frame_projection
        assert isinstance(projection, FrameProjection)
        assert projection.present is True
        assert projection.label == 0
        assert len(calls) == 1                     # exactly one lookup

        # A repeat render at the SAME selection/generation reuses the pin.
        display.update()
        assert len(calls) == 1
        assert display._current_frame_projection is projection

        # A new selection bumps the generation -> exactly one more lookup.
        display.frame_ids = (1,)
        display.update()
        assert display._current_frame_projection is not None
        assert display._current_frame_projection.label == 1
        assert len(calls) == 2
    finally:
        widget.close()
        widget.deleteLater()


def test_render_pin_absent_when_no_selection(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch, tmp_path)
    try:
        display = widget.displayframe
        display.frame_ids = ()
        display.update()
        assert display._current_frame_projection is None
    finally:
        widget.close()
        widget.deleteLater()
