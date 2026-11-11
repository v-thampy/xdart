"""A Calibrate or Import that replaces a staged row leaves no fit behind (1.2.10 review A1, Codex).

In 1.2.9 the job's end adopted the new row, saved the session with the old fit still adopted, and only then
invalidated the fit, so the session file kept a fit of control points it no longer held; reopening it adopted that
stale fit, and a failed session write skipped the invalidation. The rows use a guided session (staged frames, a real
Fit), the real Import and Open session actions and a real filesystem failure; only the file pickers are answered.
"""
from __future__ import annotations

import time

import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering.test_stitch_calibration_fit_dialog import _dispose, _twoc_fit, _wait


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _import_first_row_again(dialog, monkeypatch):
    """Import the first staged frame's own PONI and control points through the window's Import action."""
    from xdart.gui.widgets import file_dialogs

    frame = dialog.frames[0]
    picks = iter([[frame.poni_path], [frame.npt_path]])
    monkeypatch.setattr(file_dialogs, "browse", lambda *_args, **_kwargs: next(picks))
    dialog.frame_table.setCurrentCell(0, 0)
    dialog.import_button.click()
    _wait(dialog)
    return frame


def _settled(dialog, timeout=30.0):
    until = time.monotonic() + timeout
    while dialog._pending_session_resume is not None and time.monotonic() < until:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.01)
    _wait(dialog)


def test_an_imported_row_leaves_no_fit_in_the_saved_session(qapp, tmp_path, monkeypatch):
    """After Import replaces a row of a fitted session, the session it saved holds no fit, and Open session on that
    file restores the new row without the old fit."""
    from xdart.gui.widgets import file_dialogs

    page, dialog, _window = _twoc_fit(qapp, tmp_path)
    try:
        assert dialog.fit is not None
        old = _import_first_row_again(dialog, monkeypatch)
        assert dialog.frames[0] is not old and dialog.frames[0].identity == old.identity
        session = dialog.session_dir / "calibration-session.json"
        assert session.is_file(), dialog.status.text()
        monkeypatch.setattr(file_dialogs, "browse", lambda *_args, **_kwargs: [str(session)])
        dialog.open_session_button.click()
        _settled(dialog)
        assert dialog.fit is None, dialog.status.text()
        assert not dialog.save_button.isEnabled()
        assert dialog.frames[0].poni_sha256 == old.poni_sha256
    finally:
        dialog.close()
        _dispose(page)


def test_a_failed_session_write_after_an_import_still_retires_the_fit(qapp, tmp_path, monkeypatch):
    """When the session cannot be written after Import replaced a row, the fit of the old control points is retired
    all the same (Save is off), and the window shows the new row and why the session was not saved."""
    page, dialog, _window = _twoc_fit(qapp, tmp_path)
    try:
        assert dialog.fit is not None
        (dialog.session_dir / "calibration-session.tmp").mkdir()  # the session write's temporary is a folder
        old = _import_first_row_again(dialog, monkeypatch)
        assert dialog.fit is None, dialog.status.text()
        assert not dialog.save_button.isEnabled()
        assert dialog.frames[0] is not old and dialog.frames[0].identity == old.identity
        assert dialog.status.text(), "the failed write says why"
    finally:
        dialog.close()
        _dispose(page)
