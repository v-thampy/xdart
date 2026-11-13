"""C-5 of NEXT-1210-RESPONSIVENESS: opening a calibration session runs on the window's worker.

The plan (handoffs/review_fixes_1210_2026-11-11.md, C-5, from the 1.2.9 review's A3): opening a session decoded and
hashed up to 32 staged images on the GUI thread. The same checks run on the window's existing SingleWorkerOwner, with
its busy state and Cancel; the input integrity checks stay; and the image digest reads the pixels without the extra
full copy, giving the same digest every saved session recorded.
"""
from __future__ import annotations

import hashlib
import threading
import time
from threading import Event, Timer

import numpy as np
import pytest

from tests.xdart.scattering.test_stitch_calibration_fit_dialog import (  # noqa: F401 (qapp)
    _guided_frames, _page_with_dialog, _wait, qapp,
)
from tests.xdart.scattering.test_stitch_calibration_bounds import _dispose


def test_opening_a_session_runs_on_the_window_worker(qapp, tmp_path, monkeypatch):
    from xrd_tools.integrate import stitch_calibration
    from xrd_tools.integrate.stitch_calibration import save_calibration_session
    (tmp_path / 'frames').mkdir()
    frames = _guided_frames(tmp_path)
    page, dialog = _page_with_dialog(tmp_path)
    let_go = None
    try:
        (tmp_path / 'session-a').mkdir()
        path = save_calibration_session(tmp_path / 'session-a', frames, arm=('del', 'nu'),
                                        fit_policy=dialog._fit_policy())
        entered, release, on_gui = Event(), Event(), []
        real = stitch_calibration.validate_frame_files

        def held(frame, **kwargs):  # a staged file's hash, held as a slow share would hold it
            on_gui.append(threading.current_thread() is threading.main_thread())
            if not entered.is_set():
                entered.set()
                release.wait(10)
            return real(frame, **kwargs)
        monkeypatch.setattr(stitch_calibration, 'validate_frame_files', held)
        monkeypatch.setattr('xdart.gui.widgets.file_dialogs.browse', lambda *args, **kwargs: [str(path)])
        let_go = Timer(3.0, release.set)  # where the GUI thread does the checks, it is let go so the row ends
        let_go.start()
        started = time.monotonic()
        dialog._resume()
        took = time.monotonic() - started
        busy = dialog.active() and dialog.cancel_button.isEnabled()
        release.set()
        assert took < 1.5, f"opening a session held the window {took:.2f} s"  # the parent held it 3 s
        assert busy  # its busy state, with Cancel
        _wait(dialog)
        assert len(dialog.frames) == len(frames), dialog.status.text()
        assert on_gui and not any(on_gui)  # every check ran off the GUI thread
    finally:
        if let_go is not None:
            let_go.cancel()
        _dispose(page)


def _identity_before(image):
    """calibration_image_identity as 1.2.9 computed it (a full float64 copy, then its bytes): the digest every saved
    session records."""
    values = np.array(np.asarray(image), dtype='<f8', order='C', copy=True)
    values[np.isnan(values)] = np.nan
    shape = tuple(int(v) for v in values.shape)
    digest = hashlib.sha256(b'xdart-calibration-image-float64-v1\0')
    digest.update(np.asarray(shape, dtype='<u8').tobytes())
    digest.update(values.tobytes())
    return shape, digest.hexdigest()


def _odd_nan():
    values = np.arange(12, dtype='<f8').reshape(3, 4)
    raw = values.view('<u8')
    raw[1, 2] = 0x7FF8000000000ABC  # a NaN with a non-canonical payload
    raw[2, 0] = 0xFFF0000000000001  # a signalling NaN with the sign set
    return values


@pytest.mark.parametrize('image', [
    np.arange(20, dtype=np.uint16).reshape(4, 5),
    np.arange(20, dtype=np.int32).reshape(4, 5) - 7,
    np.where(np.arange(20).reshape(4, 5) % 3, np.arange(20.).reshape(4, 5), np.nan).astype(np.float32),
    _odd_nan(),
    np.arange(20, dtype='>f8').reshape(4, 5),
    np.asfortranarray(np.arange(20.).reshape(4, 5)),
    np.arange(40.).reshape(4, 10)[:, ::2],
], ids=['uint16', 'int32', 'float32 NaN', 'float64 NaN payloads', 'big-endian', 'Fortran', 'strided'])
def test_the_image_digest_is_unchanged_without_the_extra_copy(image):
    from xrd_tools.integrate.stitch_calibration import calibration_image_identity
    original = np.array(image, copy=True)
    assert calibration_image_identity(image) == _identity_before(image)
    assert image.tobytes() == original.tobytes()  # the caller's pixels are untouched, NaN payloads included


@pytest.mark.parametrize('how', ['cancel', 'close'])
def test_a_cancelled_or_closed_open_applies_nothing(qapp, tmp_path, monkeypatch, how):
    """Cancel while the session is read, or closing the window then, applies nothing; Cancel leaves the window usable
    (the plan: "with the window's busy state and Cancel")."""
    from xrd_tools.integrate import stitch_calibration
    from xrd_tools.integrate.stitch_calibration import save_calibration_session
    (tmp_path / 'frames').mkdir()
    frames = _guided_frames(tmp_path)
    page, dialog = _page_with_dialog(tmp_path)
    try:
        (tmp_path / 'session-a').mkdir()
        path = save_calibration_session(tmp_path / 'session-a', frames, arm=('del', 'nu'),
                                        fit_policy=dialog._fit_policy())
        entered, release = Event(), Event()
        real = stitch_calibration.validate_frame_files

        def held(frame, **kwargs):
            if not entered.is_set():
                entered.set()
                release.wait(10)
            return real(frame, **kwargs)
        monkeypatch.setattr(stitch_calibration, 'validate_frame_files', held)
        monkeypatch.setattr('xdart.gui.widgets.file_dialogs.browse', lambda *args, **kwargs: [str(path)])
        applied, before = [], dialog.frames
        monkeypatch.setattr(dialog, '_session_opened', lambda *args: applied.append(args))
        dialog._resume()
        assert entered.wait(10)
        dialog.cancel_button.click() if how == 'cancel' else dialog.close()
        release.set()
        until = time.monotonic() + 30
        while (dialog.active() or until - time.monotonic() > 29.5) and time.monotonic() < until:
            qapp.processEvents()
            time.sleep(.01)
        assert applied == [] and dialog.frames == before, dialog.status.text()
        assert not dialog.active()
        if how == 'cancel':
            assert dialog.isVisible() and 'Cancelled' in dialog.status.text(), dialog.status.text()
        else:
            assert not dialog.isVisible()
    finally:
        release.set()
        _dispose(page)


@pytest.mark.parametrize('action', ['close', 'new', 'late cancel'])
def test_a_new_intent_discards_the_unapplied_session(qapp, tmp_path, monkeypatch, action):
    from pyqtgraph.Qt import QtWidgets
    from xrd_tools.integrate.stitch_calibration import save_calibration_session

    (tmp_path / 'frames').mkdir()
    frames = _guided_frames(tmp_path)
    page, dialog = _page_with_dialog(tmp_path)
    try:
        session = tmp_path / 'session-a'
        session.mkdir()
        path = save_calibration_session(session, frames, arm=('del', 'nu'), fit_policy=dialog._fit_policy())
        monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', staticmethod(
            lambda parent, title, start, filt: (str(path), filt)))
        before = dialog.session_dir, dialog.frames
        dialog._resume()
        if action == 'late cancel':
            worker = dialog._owner._worker
            assert worker is not None
            worker.join(10)
            assert not worker.is_alive()
            dialog.cancel_button.click()  # the result is available, but the window has not polled it
        deadline = time.monotonic() + 10
        while dialog._identity is not None and time.monotonic() < deadline:
            dialog._poll()  # consume the real worker terminal, without delivering the next-turn callback
            time.sleep(.005)
        assert dialog._identity is None
        assert (dialog.session_dir, dialog.frames) == before
        if action == 'close':
            assert dialog.close()
        elif action == 'new':
            dialog.new_session_button.click()
            before = dialog.session_dir, dialog.frames
            assert before[0] != session
        qapp.processEvents()
        assert dialog.isVisible() == (action != 'close')
        assert (dialog.session_dir, dialog.frames) == before
    finally:
        _dispose(page)
