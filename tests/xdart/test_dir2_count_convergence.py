# -*- coding: utf-8 -*-
"""DIR-2 convergence (maintainer, 2026-07-13): the 'N files' chip converges to
real frame counts two ways — LAZILY, as the run opens/retires each container
(sigContainerCount → memo), and ON DEMAND via a click on the readiness summary
(background sweep, one count_frames per not-yet-known file).  A stale file
stamp invalidates its memo entry, so a grown container never serves an old
count."""

import os
import threading as _threading
import time as _time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pyqtgraph")


@pytest.fixture(scope="module")
def qapp():
    from pyqtgraph.Qt import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _container_dir(tmp_path, n_files=3, frames=2):
    from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter
    d = tmp_path / "watch"
    d.mkdir()
    for i in range(n_files):
        _write_bluesky_nxwriter(d / f"scan_{i:05d}.nxs", n=frames)
    return d


def _point_at(widget, d):
    sig = widget.wrangler.parameters.child("Signal")
    sig.child("inp_type").setValue("Image Directory")
    sig.child("img_dir").setValue(str(d))
    sig.child("img_ext").setValue("nxs")


def test_lazy_convergence_files_to_frames(qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter

    d = _container_dir(tmp_path, n_files=3, frames=2)
    widget = staticWidget()
    try:
        _point_at(widget, d)
        assert widget._controls_v2_source_frame_count() == 3
        assert widget._v2_source_count_is_files is True

        # Counts land (as the run's sigContainerCount would deliver them).
        for i in range(3):
            widget._on_container_count_landed(str(d / f"scan_{i:05d}.nxs"), 2)
        assert widget._controls_v2_source_frame_count() == 6
        assert widget._v2_source_count_is_files is False

        # One file grows: its stamp no longer matches -> honest fallback to
        # the file count until the new count lands.
        _write_bluesky_nxwriter(d / "scan_00001.nxs", n=5)
        widget._v2_frame_count_cache = None
        assert widget._controls_v2_source_frame_count() == 3
        assert widget._v2_source_count_is_files is True
        widget._on_container_count_landed(str(d / "scan_00001.nxs"), 5)
        assert widget._controls_v2_source_frame_count() == 9
        assert widget._v2_source_count_is_files is False
    finally:
        widget.close()
        widget.deleteLater()


def test_wrangler_emits_container_counts(tmp_path):
    """The REAL directory-watch thread announces each container's frame count
    at open and its final count at retire."""
    from tests.xdart.test_bluesky_image_wrangler import (
        _real_dir_watch_thread,
        _write_bluesky_nxwriter,
    )

    watch = tmp_path / "watch"
    watch.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    p = watch / "grow_00001.nxs"
    _write_bluesky_nxwriter(p, n=3)

    t = _real_dir_watch_thread(watch, out)
    landed = []
    t.sigContainerCount.connect(lambda path, n: landed.append((path, n)))

    for _ in range(6):
        item = t._get_next_eiger_frame_sync()
        if item[3] is None:
            break
    assert (str(p), 3) in landed


def test_click_to_count_sweeps_off_gui_and_converges(qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.io import image as image_io

    d = _container_dir(tmp_path, n_files=4, frames=2)

    calls = []
    real_count = image_io.count_frames
    monkeypatch.setattr(
        image_io, "count_frames",
        lambda path: (calls.append(_threading.get_ident()),
                      real_count(path))[1])

    widget = staticWidget()
    try:
        gui_ident = _threading.get_ident()
        _point_at(widget, d)
        assert widget._controls_v2_source_frame_count() == 4  # files-mode

        # The click (files-mode) kicks the on-demand sweep...
        widget._on_readiness_summary_clicked()
        deadline = _time.monotonic() + 15
        while True:
            qapp.processEvents()
            widget._v2_frame_count_cache = None
            if (widget._controls_v2_source_frame_count() == 8
                    and not widget._v2_count_sweep_active):
                break
            assert _time.monotonic() < deadline, "sweep never converged"
            _time.sleep(0.01)
        assert widget._v2_source_count_is_files is False
        # ...entirely off the GUI thread.
        assert calls and all(ident != gui_ident for ident in calls)

        # A second click with everything memoized counts nothing new.
        calls.clear()
        widget._on_readiness_summary_clicked()
        qapp.processEvents()
        assert calls == []
    finally:
        widget.close()
        widget.deleteLater()


def test_files_mode_tooltip_hint_and_source_header_unit(qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xdart.gui.tabs.static_scan.controls_logic import build_control_profile
    from xdart.gui.tabs.static_scan.ui.controls_panel_v2 import ControlsPanelV2

    d = _container_dir(tmp_path, n_files=2, frames=2)
    widget = staticWidget()
    try:
        _point_at(widget, d)
        widget._refresh_controls_v2_profile(immediate=True)
        # Click affordance: the tooltip advertises the on-demand count.
        tip = widget.controls.readinessLabel.toolTip()
        assert "Click to count frames" in tip

        # The source-card header renders 'files' for a container directory.
        state = widget._controls_v2_state()
        assert state.frame_count_is_files is True
        profile = build_control_profile(state)
        header = ControlsPanelV2._source_status(profile)
        assert "2 files" in header
        assert "frames" not in header
    finally:
        widget.close()
        widget.deleteLater()
