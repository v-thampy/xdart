"""A Calibrate or Import that replaces a staged row leaves no fit behind (1.2.10 review A1, Codex).

In 1.2.9 the job's end adopted the new row, saved the session with the old fit still adopted, and only then
invalidated the fit, so the session file kept a fit of control points it no longer held; reopening it adopted that
stale fit, and a failed session write skipped the invalidation. The rows use a guided session (staged frames, a real
Fit), the real Import and Open session actions and a real filesystem failure; only the file pickers are answered.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering.test_stitch_calibration_fit_dialog import _dispose, _twoc_fit, _wait
from xrd_tools.integrate.stitch_calibration import load_calibration_session


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


def test_an_imported_row_leaves_no_fit_in_the_saved_session(qapp, tmp_path, monkeypatch):
    """After Import replaces a row of a fitted session, the session it saved holds no fit, and Open session on that
    file restores the new row without the old fit."""
    from xdart.gui.widgets import file_dialogs

    page, dialog, _window = _twoc_fit(qapp, tmp_path)
    try:
        assert dialog.fit is not None
        old = _import_first_row_again(dialog, monkeypatch)
        replacement = dialog.frames[0]
        assert replacement.identity == old.identity
        assert replacement.poni_path != old.poni_path and replacement.npt_path != old.npt_path
        session = dialog.session_dir / "calibration-session.json"
        saved, _, _, resume = load_calibration_session(session, with_resume=True)
        assert resume is not None and resume.fit is None and resume.fit_source is None, resume
        assert (saved[0].poni_path, saved[0].npt_path) == (replacement.poni_path, replacement.npt_path)
        monkeypatch.setattr(file_dialogs, "browse", lambda *_args, **_kwargs: [str(session)])
        dialog.open_session_button.click()
        _wait(dialog)
        assert dialog.fit is None, dialog.status.text()
        assert not dialog.save_button.isEnabled()
        restored = dialog.frames[0]
        assert restored is not replacement
        assert (restored.poni_path, restored.npt_path) == (replacement.poni_path, replacement.npt_path)
    finally:
        dialog.close()
        _dispose(page)


def test_a_failed_session_write_after_an_import_still_retires_the_fit(qapp, tmp_path, monkeypatch):
    """When the session cannot be written after Import replaced a row, the fit of the old control points is retired
    all the same (Save is off), and the window shows the new row and why the session was not saved."""
    page, dialog, _window = _twoc_fit(qapp, tmp_path)
    try:
        assert dialog.fit is not None
        session = dialog.session_dir / "calibration-session.json"
        saved = session.read_bytes()
        (dialog.session_dir / "calibration-session.tmp").mkdir()  # the session write's temporary is a folder
        old = _import_first_row_again(dialog, monkeypatch)
        assert dialog.fit is None, dialog.status.text()
        assert not dialog.save_button.isEnabled()
        replacement = dialog.frames[0]
        assert replacement.identity == old.identity and replacement.poni_path != old.poni_path
        assert dialog.frame_table.item(0, 3).text() == Path(replacement.poni_path).name
        assert session.read_bytes() == saved
        assert "calibration-session.tmp" in dialog.status.text(), dialog.status.text()
    finally:
        dialog.close()
        _dispose(page)
