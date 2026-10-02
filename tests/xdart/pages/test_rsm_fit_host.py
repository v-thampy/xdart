"""The native page supplies admitted artifacts to the shared fit windows."""
import time

import numpy as np

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.rsm.page import RSMPage
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.notebooks import load_snapshot
from tests.xdart.pages.test_rsm_page import _Status
from tests.xdart.scattering.test_rsm_fit_windows import payload, qapp  # noqa: F401


def settle(page, qapp):
    end = time.monotonic() + 30
    while page.active() and time.monotonic() < end:
        qapp.processEvents()
        time.sleep(.01)
    qapp.processEvents()
    assert not page.active()


def dispose(page, qapp):
    end = time.monotonic() + 30
    receipt = page.close_page()
    while receipt.status is PageCleanup.PENDING and time.monotonic() < end:
        qapp.processEvents()
        time.sleep(.01)
        receipt = page.close_page()
    assert receipt.status is PageCleanup.CLEAN
    page.deleteLater()
    qapp.processEvents()


def test_page_mounts_native_lineout_plane_and_volume_fits(qapp, tmp_path):
    page = RSMPage(empty_host_services(_Status()))
    admitted = payload(tmp_path)
    path = admitted.inspection.path
    try:
        page._rows['project_root'].editor.setText(str(tmp_path))
        page._add_result(path, admitted)
        for name in ('lineout', 'fit2d', 'fit3d'):
            page.tools.page_tool(name).click()
            settle(page, qapp)
            window = page._tool_windows[name]
            assert window.current().reference.path == path
            captured = window.current().load()
            replay = load_snapshot(window.current().reference, bundle_root=tmp_path)
            np.testing.assert_array_equal(captured.values, replay.values)
            if name == 'lineout':
                assert window.host.source_root == str(tmp_path)
                assert window.fit_dialog.batch_btn.isEnabled()
                window.fit_dialog.fit_btn.click()
                settle(page, qapp)
                assert window.host.outcomes[-1].ok, window.fit_dialog.status.text()
            else:
                assert window.source_root == str(tmp_path)
                window.fit_button.click()
                settle(page, qapp)
                assert window.outcome is not None, window.status.text()
        outcome = page._tool_windows['fit3d'].outcome
        page._view_command(ShellCommand(ShellCommandKind.SET_LOG_SCALE, True))
        settle(page, qapp)
        assert page._tool_windows['fit3d'].outcome is outcome
        page._follow_tools()
        assert page._tool_windows['fit3d'].outcome is outcome
    finally:
        dispose(page, qapp)


def test_lineout_batch_is_lazy_project_scoped_and_keeps_scalar_rows(qapp, tmp_path):
    page = RSMPage(empty_host_services(_Status()))
    try:
        page._rows['project_root'].editor.setText(str(tmp_path))
        (tmp_path/'one').mkdir()
        (tmp_path/'two').mkdir()
        first, second = payload(tmp_path/'one'), payload(tmp_path/'two')
        page._add_result(first.inspection.path, first, show=False)
        page._add_result(second.inspection.path, second)
        page.tools.page_tool('lineout').click()
        settle(page, qapp)
        window = page._tool_windows['lineout']
        # No selection: Batch uses all listed results, including one that is no longer resident.
        page._results[first.inspection.path] = None
        page.browser.clear_result_selection()
        targets = window.batch()
        assert [target.label for target in targets] == [first.inspection.path, second.inspection.path]
        np.testing.assert_array_equal(targets[0].load().values, window.current().load().values)
        missing = targets[0].trend('unavailable_column')
        assert not missing.available
        window.fit_dialog.batch_btn.click()
        settle(page, qapp)
        assert len(window.host.outcomes) == 2
        assert all(outcome.ok for outcome in window.host.outcomes)
        assert window.host.outcomes[0].overlay is None
        assert window.host.outcomes[-1].overlay is not None
        before = window.scope()
        page._listed.pop(0)
        assert window.scope() != before
    finally:
        dispose(page, qapp)
