"""Stitch keeps only the results it shows in memory (STITCH-2D-20260930 C12; Codex general audit AUD-5,
narrowed by architecture invariant 12: a processed file can always be read or stitched again). Measured on
LaB6 scan 14 (287 frames): a Stitch 2D result holds 4.65 MB, a Stitch 1D 0.32 MB; every result stayed
until the page closed. The others are listed by name and reloaded from their files when shown again.
"""
from pathlib import Path

from pyqtgraph.Qt import QtCore

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.stitch.page import StitchPage
from tests.xdart.pages.test_stitch_page import _Status, _wait, qapp  # noqa: F401
from tests.xdart.tools.test_stitch_native_values import _native_form


def test_only_the_shown_results_stay_in_memory_and_the_others_reload(qapp, tmp_path):
    page = StitchPage(empty_host_services(_Status()))
    select = QtCore.QItemSelectionModel.SelectionFlag
    role = QtCore.Qt.ItemDataRole.UserRole
    try:
        page.load_form(_native_form(tmp_path, "multigeometry"))  # three outputs
        page.run_bar.startButton.click()
        _wait(page, qapp)
        paths = [page.results.item(row).data(role) for row in range(page.results.count())]
        assert len(paths) == 3 and all(Path(path).is_file() for path in paths)
        # Single: the current result only (the last one stitched)
        assert set(page._payloads) == {paths[-1]}
        page.results.setCurrentRow(0)
        _wait(page, qapp)
        assert set(page._payloads) == {paths[0]} and page.frames.count() == 2
        # Overlay: the selected results
        page._view_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Overlay"))
        page.results.setCurrentRow(2, select.ClearAndSelect)
        page.results.setCurrentRow(0, select.Select)
        _wait(page, qapp)
        assert set(page._payloads) == {paths[0], paths[2]}
        assert len(page.curve.listDataItems()) == 2
        # a result whose file is gone stays listed, with the reason, and holds nothing
        Path(paths[1]).unlink()
        page.results.setCurrentRow(1, select.ClearAndSelect)
        _wait(page, qapp)
        assert page.results.count() == 3 and paths[1] not in page._payloads
        status = page.run_bar.readinessLabel.full_text()
        assert Path(paths[1]).name in status and "No such file" in status, status
        assert not page.run_bar.readinessDot.property("ready")
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()
