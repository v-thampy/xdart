"""Switching Live on says how to browse the scans a Run has processed (maintainer, 2026-10-26: "it will also be good
to have a little message when live mode is toggled saying that the run should be paused/stopped to browse already
processed scans"). The real page and its shell commands."""
from __future__ import annotations

import pytest

pytest.importorskip("pyqtgraph")

from tests.xdart.scattering.test_browse_reference_rings_run import _directory, _page
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind

NOTE = "Live is on: pause or stop the Run to browse the scans it has already processed."


@pytest.fixture
def qapp(_xdart_qt_harness):
    from pyqtgraph.Qt import QtWidgets
    return QtWidgets.QApplication.instance()


def test_switching_live_on_says_to_pause_or_stop_the_run_to_browse(qapp, tmp_path) -> None:
    with _page(qapp, _directory(tmp_path, live=False)) as (page, _scientific, _events):
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_LIVE, True))
        assert page._intents.snapshot().thaw().live_mode and page._notice_text == NOTE
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_LIVE, False))
        assert not page._intents.snapshot().thaw().live_mode and page._notice_text != NOTE
