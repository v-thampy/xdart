"""A Run that holds the page's reads says so (beamline 2026-10-25).

The Metadata window opened while a Live Run was paused stayed blank until the Run stopped ("say why it's empty for
now, but i want to revisit this"), and a scan clicked in the Browser while a Live Run was running logged
"Browse refused" with a traceback. Real pages driven through their real lifecycle owner.
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.events import DurablePaused, PauseRequested
from xdart.gui.tabs.scattering.state_machine import RunPhase
from tests.xdart.scattering.test_e1b2_page_command_boundaries import (
    _Executor,
    _active_page,
    _dispose,
    _paced_frame_events,
)


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_metadata_window_says_why_it_is_empty_while_a_run_is_paused(qapp) -> None:
    executor = _Executor()
    page, lifecycle, identity = _active_page(executor)
    try:
        executor.events.extend(_paced_frame_events(page, executor, identity, 1))
        page._drain_executor()
        assert page._context_controller.navigation.current is not None
        lifecycle.pause_requested(PauseRequested(identity))
        lifecycle.durable_paused(DurablePaused(identity, 1))
        assert lifecycle.phase is RunPhase.PAUSED

        page._open_analysis_mount("metadata")

        dialog = page._metadata_dialog
        assert dialog is not None and dialog.table.rowCount() == 0
        assert dialog.status.text() == (
            "Metadata is available after the Run stops; then press Metadata again.")
        assert not dialog.plot_button.isEnabled()
    finally:
        _dispose(page, qapp)


def test_browse_while_a_run_is_running_is_a_notice_not_an_error(qapp, monkeypatch, caplog) -> None:
    executor = _Executor()
    page, lifecycle, _identity = _active_page(executor)
    notices: list[str] = []
    monkeypatch.setattr(page, "_notice", notices.append)
    try:
        assert lifecycle.phase is RunPhase.RUNNING
        with caplog.at_level(logging.ERROR):
            page._select_scan("/data/xdart_processed_data/scan_int2d.nexus", reopen=True)

        assert notices[-1] == ("Browse is unavailable while a Run is running; pause or stop it to open another "
                               "scan.")
        assert "Browse refused" not in caplog.text and "Traceback" not in caplog.text
        assert page._context_controller.browse_context is None and not page._context_controller.browse_pending
    finally:
        _dispose(page, qapp)
