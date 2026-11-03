"""Item 11 review: invalid drafts never fit a different frame; read refusals reach a popup."""

import pytest
from pyqtgraph.Qt import QtWidgets

from tests.core.test_int_fit_source import _result, _peak, _stored
from tests.xdart.test_fit_source_gui import _choose_nexus, _displayed, _drain, _window
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


def test_raising_a_file_source_window_does_not_replace_it_with_ints_trace(qapp, tmp_path, monkeypatch):
    import numpy as np
    main = _result(tmp_path / "main.nexus")
    chosen = _result(tmp_path / "chosen.nexus", shift=.3)
    page = _page()
    try:
        trace, anchor = _displayed(main, 2)
        monkeypatch.setattr(page, "_capture_analysis_trace", lambda _kind: (trace, anchor))
        dialog, host = _window(page, "peak")
        _choose_nexus(dialog, chosen, 1)
        _drain(qapp, host)
        np.testing.assert_array_equal(dialog._y, _stored(_peak(2.4)))
        page._open_analysis_mount("peak")  # the same launcher raises its retained window
        assert page._peak_dialog is dialog and dialog.fit_source().path == str(chosen)
        np.testing.assert_array_equal(dialog._y, _stored(_peak(2.4)))
    finally:
        _close_page(page, qapp)


def test_a_fit_time_source_read_problem_also_opens_its_popup(qapp, tmp_path):
    path = _result(tmp_path / "sample.nexus")
    page = _page()
    try:
        dialog, host = _window(page, "peak")
        _choose_nexus(dialog, path, 1)
        _drain(qapp, host)
        path.rename(tmp_path / "moved.nexus")
        dialog.fit_btn.click()
        _drain(qapp, host)
        assert host.outcomes and not host.outcomes[-1].ok
        box = dialog.source_row.message_box
        assert box is not None and box.isVisible(), dialog.status.text()
        assert "FIT_SOURCE_UNREADABLE" in box.informativeText()
    finally:
        _close_page(page, qapp)
