"""The analysis display anchor on a real page: a fit adopts its result only while Int still shows the frame and
context it was captured from (1.2.10 review M6: the guard's refusals ran only on SimpleNamespace doubles, and six
tests patch it to always-True; this row is in place before any of those patches is removed)."""
from __future__ import annotations

import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering.test_e4_accumulation_selection_parity import _click_frame
from tests.xdart.scattering.test_e4_preview_transport import _write_processed
from tests.xdart.scattering.test_p3_experiment_operation_composition import _page
from tests.xdart.scattering.test_p34_reintegrate_operation import _wait
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_display_anchor_refuses_another_frame_file_fingerprint_or_dialog_generation(tmp_path, monkeypatch, qapp):
    import xdart.gui.tabs.scattering.analysis_mount as mount

    first, _ = _write_processed(tmp_path / "first", labels=(1, 2))
    second, _ = _write_processed(tmp_path / "second", labels=(1, 2))
    page, _ = _page(tmp_path, monkeypatch,
                    store=RunIntentStore(RunIntent(project_root=str(tmp_path), processing_mode="Int 1D")))
    controller, shell = page._context_controller, page._shell

    seen = []

    def shown(path, label):
        frame = controller.navigation.current
        state = (None if frame is None else (frame.artifact == str(path), frame.local_frame_label),
                 controller.browse_pending or page._scientific_repaint_pending or shell.browser.frame_selection_pending,
                 mount.capture_display_anchor(page, "probe", 0) is not None)
        seen[:] = [state]
        return True if state == ((True, label), False, True) else None

    def until_shown(path, label):
        try:
            _wait(lambda: shown(path, label))
        except AssertionError as error:
            raise AssertionError(f"not shown: last (frame, browse/repaint/selection pending, anchor) = {seen}") from error

    def open_file(path):
        page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_SCAN, str(path), path=("artifact",)))
        until_shown(path, 1)

    try:
        page.resize(1200, 800)
        page.show()
        open_file(first)
        anchor = mount.capture_display_anchor(page, "trace-fp", 3)
        assert anchor is not None and anchor.controller is controller
        assert mount.display_anchor_matches(page, anchor, "trace-fp", 3)
        assert not mount.display_anchor_matches(page, anchor, "other-fp", 3)
        assert not mount.display_anchor_matches(page, anchor, "trace-fp", 4)

        _click_frame(qapp, shell, 1)
        until_shown(first, 2)
        assert not mount.display_anchor_matches(page, anchor, "trace-fp", 3), "another frame is shown"

        again = mount.capture_display_anchor(page, "trace-fp", 3)
        assert again is not None and mount.display_anchor_matches(page, again, "trace-fp", 3)
        open_file(second)
        assert not mount.display_anchor_matches(page, again, "trace-fp", 3), "another file's context is shown"
    finally:
        _wait(lambda: True if page.close_workspace().cleanup_status is CleanupStatus.CLEANED else None)
        page.deleteLater()
        qapp.processEvents()
