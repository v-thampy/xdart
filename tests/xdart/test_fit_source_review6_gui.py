"""Item 11 review: invalid drafts never fit a different frame; read refusals reach a popup."""

import pytest
from pyqtgraph.Qt import QtWidgets

from tests.core.test_int_fit_source import _result
from tests.xdart.test_fit_source_gui import _choose_nexus, _drain, _window
from tests.xdart.scattering.test_fit_tools_preservation import _close_page, _page


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_an_invalid_source_frame_never_fits_the_previous_frame(qapp, tmp_path):
    path = _result(tmp_path / "sample.nexus")
    page = _page()
    try:
        dialog, host = _window(page, "peak")
        _choose_nexus(dialog, path, 1)
        _drain(qapp, host)
        editor = dialog.source_row.frame_row.editor
        editor.setText("not a frame")
        editor.editingFinished.emit()
        dialog.fit_btn.click()
        _drain(qapp, host)
        assert not host.outcomes, "invalid frame text must not silently fit the previous frame"
        assert "FIT_SOURCE_INVALID" in dialog.status.text(), dialog.status.text()
    finally:
        _close_page(page, qapp)


def test_a_source_read_problem_opens_once_in_a_popup(qapp, tmp_path):
    path = _result(tmp_path / "sample.nexus")
    page = _page()
    try:
        dialog, host = _window(page, "peak")
        _choose_nexus(dialog, path, 999)
        _drain(qapp, host)
        row = dialog.source_row
        assert row.message_box is not None and row.message_box.isVisible()
        assert "FIT_SOURCE_FRAME_MISSING" in row.message_box.informativeText()
        box = row.message_box
        dialog.refresh_btn.click()
        _drain(qapp, host)
        assert row.message_box is box, "the same source problem does not repeat"
    finally:
        _close_page(page, qapp)
