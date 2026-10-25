"""A running Run grays out the Browser's scan list (maintainer, 2026-10-25).

"For a live run where browsing is disabled, it will be good to make it explicit by disabling and graying out the
data browser altogether until paused. There can be a tooltip to make it clear why." His pick: "Scan list only" (the
running scan's frames stay usable). The shared BrowserView owner, then a real page through its lifecycle owner.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.browser_view import BrowserView
from xdart.gui.tabs.scattering.events import DurablePaused, PauseRequested
from xdart.gui.tabs.scattering.shell_values import BrowserProjection, FrameNavigationProjection
from xdart.gui.tabs.scattering.state_machine import RunPhase
from tests.xdart.scattering.test_e1b2_page_command_boundaries import _Executor, _active_page, _dispose


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _catalog_widgets(view: BrowserView):
    return (view.scans, view.show_dirs, view.date_sort, view.refresh, view.filename_filter)


def test_browser_view_grays_its_scan_list_with_the_reason_and_restores_it(qapp) -> None:
    view = BrowserView()
    try:
        own = [widget.toolTip() for widget in _catalog_widgets(view)]
        navigation = FrameNavigationProjection()
        view.reconcile(BrowserProjection(scans_locked="A Run is running."), navigation, plot_mode="Single")
        assert not any(widget.isEnabled() for widget in _catalog_widgets(view))
        assert all(widget.toolTip() == "A Run is running." for widget in _catalog_widgets(view))
        assert view.frames.isEnabled()

        view.reconcile(BrowserProjection(), navigation, plot_mode="Single")
        assert all(widget.isEnabled() for widget in _catalog_widgets(view))
        assert [widget.toolTip() for widget in _catalog_widgets(view)] == own
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_a_running_run_locks_the_scan_list_until_it_is_paused(qapp) -> None:
    executor = _Executor()
    page, lifecycle, identity = _active_page(executor)
    try:
        browser = page._shell.browser
        page._refresh_shell()
        assert lifecycle.phase is RunPhase.RUNNING
        assert not browser.scans.isEnabled() and browser.frames.isEnabled()
        assert browser.scans.toolTip().startswith("Other scans open while the Run is paused or after it stops.")

        lifecycle.pause_requested(PauseRequested(identity))
        lifecycle.durable_paused(DurablePaused(identity, 1))
        page._refresh_shell()
        assert lifecycle.phase is RunPhase.PAUSED
        assert browser.scans.isEnabled() and not browser.scans.toolTip()
    finally:
        _dispose(page, qapp)
