"""Terminal adapter presentation; backend admission has separate real-data tests."""
import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering.test_workspace_operations import _capture, _stamp
from tests.xdart.scattering.test_p3_experiment_operation_composition import _page, _close
from xdart.gui.tabs.scattering.operation_values import (
    OperationIdentity, OperationTerminal, OperationTerminalStatus, OperationUpdate,
)
from xdart.gui.tabs.scattering.workspace_operations import (
    ReintegrateOperationState, WorkspaceRefreshEffect,
)


@pytest.mark.parametrize("status", [OperationTerminalStatus.FAILED,
                                   OperationTerminalStatus.CANCELLED])
def test_terminal_reintegrate_notice_reaches_footer_without_plot_repaint(tmp_path, monkeypatch, status):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page, _ = _page(tmp_path, monkeypatch)
    try:
        page._notice("Reintegrating 1-D into a replacement candidate…")
        page._refresh_shell()
        capture, identity = _capture(), OperationIdentity(1)
        operations = page._workspace_operations
        operations._reintegrate = ReintegrateOperationState(identity, capture, "1d", _stamp(capture))
        # The worker slot retires before its terminal update reaches the page.
        assert operations.current_identity is None
        update = OperationUpdate(identity, terminal=OperationTerminal(
            identity, status, diagnostic=("saved-file admission refused"
                if status is OperationTerminalStatus.FAILED else ""),
        ))
        effect = page._consume_reintegrate_update(update)
        assert effect is WorkspaceRefreshEffect.CONTROLS
        page._refresh_shell(preserve_display=True, preserve_scientific=True)
        assert not operations.owned
        assert page._shell.scientific.status.text() == page._notice_text
        assert "Reintegrating" not in page._notice_text
        # Repeated/foreign terminal updates cannot repaint the footer.
        assert page._consume_reintegrate_update(update) is WorkspaceRefreshEffect.NONE
        assert page._shell.scientific.status.text() == page._notice_text
    finally:
        _close(page, app)
