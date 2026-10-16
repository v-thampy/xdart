"""GUI-PERF-02 (GUI-RESPONSIVENESS-PACKET-B-20261016): the calibration window predicts its staged frames' rings
off the GUI thread.

The frame table appears at once; every row's rings arrive together from one worker for the window's current inputs;
a superseded edit's rings are never shown; the rings neither wait for nor hold the calibration lane (its Fit,
Calibrate and predict jobs); the window's close receipt covers their worker, and nothing arrives after it. The rings
are the headless kernel's (``predict_frame_seed``), exactly: per-frame energy, geometry, masks and invalid pixels.

``_KernelStarts`` observes which threads start the prediction kernels (``sys.monitoring`` PY_START); it replaces
nothing.
"""
from dataclasses import replace
import os
from pathlib import Path
import sys
import threading
import time

import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
from xrd_tools.integrate import calibration_seed, stitch_calibration
from xrd_tools.integrate.calibration_seed import predict_frame_seed, ring_summary

#: A row whose rings are still being predicted.
PENDING = '…'
_KERNELS = (calibration_seed.predict_frame_seed, calibration_seed.expected_rings,
            stitch_calibration.calibration_frame_mask)


class _KernelStarts:
    """(kernel name, thread ident) for every start of a prediction kernel while active."""

    def __enter__(self):
        self.tool = next(tool for tool in range(6) if sys.monitoring.get_tool(tool) is None)
        sys.monitoring.use_tool_id(self.tool, 'calibration-ring-oracle')
        self.starts = []
        sys.monitoring.register_callback(self.tool, sys.monitoring.events.PY_START,
                                         lambda code, _offset: self.starts.append((code.co_name, threading.get_ident())))
        for kernel in _KERNELS:
            sys.monitoring.set_local_events(self.tool, kernel.__code__, sys.monitoring.events.PY_START)
        return self

    def __exit__(self, *_exc):
        for kernel in _KERNELS:
            sys.monitoring.set_local_events(self.tool, kernel.__code__, 0)
        sys.monitoring.register_callback(self.tool, sys.monitoring.events.PY_START, None)
        sys.monitoring.free_tool_id(self.tool)

    def count(self, name, *, gui):
        main = threading.main_thread().ident
        return sum(1 for kernel, ident in self.starts if kernel == name and (ident == main) == gui)


def _cells(dialog):
    return [(dialog.frame_table.item(row, 5).text(), dialog.frame_table.item(row, 5).toolTip())
            for row in range(dialog.frame_table.rowCount())]


def _settle(dialog, seen=None, seconds=120):
    """Run the event loop (sleeping, as the app idles: qWait holds the GIL from the worker) until no row's rings are
    pending; ``seen`` collects every table state shown meanwhile."""
    until = time.monotonic() + seconds
    while True:
        QtWidgets.QApplication.processEvents()
        cells = _cells(dialog)
        if seen is not None:
            seen.append(cells)
        if all(text != PENDING for text, _ in cells) or time.monotonic() > until:
            return cells
        time.sleep(.005)


def _wait_idle(dialog, seconds=120):
    until = time.monotonic() + seconds
    while dialog.active() and time.monotonic() < until:
        QtWidgets.QApplication.processEvents()
        time.sleep(.005)
    assert not dialog.active(), dialog.status.text()


def _expected(dialog, frames):
    """Each row's cell from the headless kernel itself, on the window's current inputs and the calibration the parent
    predicts from (this session's fit, else its arm prediction, the rough geometry, then a geometry file)."""
    seed = dialog._seed_options()
    record = next(value for value in (seed['record'], seed['prediction'], seed['rough'], seed['file'])
                  if value is not None)
    band = dialog._conditioning()
    saturation = dialog._saturation_value() if band.mode != 'off' else None
    cells = []
    for frame in frames:
        rings = predict_frame_seed(record, frame, calibrant=seed['calibrant'], del_motor=seed['del_motor'],
                                   nu_motor=seed['nu_motor'], mask=dialog._loaded_mask(), conditioning=band,
                                   saturation=saturation).rings
        cells.append((ring_summary(rings) or 'none',
                      '\n'.join(f'Ring {ring.number} · 2θ {ring.two_theta_deg:.2f}° · d {ring.d_spacing_A:.4f} Å'
                                for ring in rings) or 'No calibrant ring reaches this frame.'))
    return cells


def _staged(tmp_path):
    """Four real staged Pilatus 300K-W frames, the first calibrated (its PONI and points adopted)."""
    from tests.core.test_stacked_arm_extension import _calibration, _ring_controls
    from tests.core.test_stitch_guided_calibration import _frames, _turn_controls
    truth = {'nu_offset': -.02, 'del_offset': .05}
    _unused, poni = _turn_controls((), _calibration().poni)
    calibration = replace(_calibration(), poni=poni)
    controls, _calibrant = _ring_controls(truth, calibration, [(5., -10.), (45., -10.), (25., 9.5), (5., 29.)])
    (tmp_path / 'frames').mkdir()
    calibrated = _frames(tmp_path / 'frames', controls, poni)
    waiting = tuple(replace(frame, poni_path='', poni_sha256='', npt_path='', npt_sha256='', pairing_policy='',
                            control_image_sha256='') for frame in calibrated[1:])
    return (calibrated[0], *waiting), poni


def _window(tmp_path, poni):
    """The window on a typed rough geometry (it predicts every frame), the frames not yet shown."""
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('manual'))
    for key, value in (('rough_distance', '390'), ('beam_pixel', f'{poni.poni2 / 172e-6:.1f}, {poni.poni1 / 172e-6:.1f}')):
        dialog._set_field_text(key, value)
    dialog.session_dir = tmp_path / 'frames'
    return dialog


def _close(dialog):
    until = time.monotonic() + 60
    receipt = dialog.shutdown()
    while receipt.status is PageCleanup.PENDING and time.monotonic() < until:
        time.sleep(.01)
        receipt = dialog.shutdown()
    dialog.close()
    return receipt


def test_the_table_appears_at_once_and_every_frames_rings_come_from_the_worker(tmp_path, request):
    frames, poni = _staged(tmp_path)
    dialog = _window(tmp_path, poni)
    try:
        dialog.frames = frames
        with _KernelStarts() as starts:
            started = time.perf_counter()
            dialog._render_frames()
            render_s = time.perf_counter() - started
            first = _cells(dialog)
            cells = _settle(dialog)
        request.node.user_properties.append(('render_gui_s', f'{render_s:.6f}'))
        expected = _expected(dialog, frames)
        assert all(text not in ('—', 'none', PENDING) for text, _ in expected)  # every frame has rings to show
        # parent polarity: the parent predicts every frame here, synchronously on the GUI thread
        assert starts.count('predict_frame_seed', gui=True) == 0
        assert starts.count('calibration_frame_mask', gui=True) == 0
        assert starts.count('expected_rings', gui=True) == 0
        assert [text for text, _ in first] == [PENDING] * len(frames)  # the basic table first
        assert all(tip for _, tip in first)  # each pending cell explains itself
        assert cells == expected  # exactly the kernel's rings, every row
        assert starts.count('predict_frame_seed', gui=False) == len(frames)  # once per frame, on the worker
        # unchanged inputs show the rings shown at once and predict nothing (the parent's cache hits)
        with _KernelStarts() as again:
            dialog._render_frames()
            assert _cells(dialog) == expected
        assert again.starts == []
        assert _close(dialog).status is PageCleanup.CLEAN
    finally:
        dialog.close()


def test_a_superseding_edit_shows_only_the_latest_rings(tmp_path):
    frames, poni = _staged(tmp_path)
    dialog = _window(tmp_path, poni)
    try:
        dialog.frames = frames
        dialog._render_frames()  # LaB6's rings asked for
        first = _expected(dialog, frames)
        dialog.refine_calibrant.setCurrentText('CeO2')  # its edit re-renders: CeO2 supersedes LaB6 before any poll
        seen = []
        cells = _settle(dialog, seen)
        latest = _expected(dialog, frames)
        assert cells == latest and latest != first
        superseded = {cell for cell in first if cell not in latest}
        assert superseded and not any(cell in superseded for state in seen for cell in state)  # never shown
        assert _close(dialog).status is PageCleanup.CLEAN
    finally:
        dialog.close()


def test_the_rings_neither_wait_for_nor_hold_the_calibration_lane(tmp_path):
    frames, poni = _staged(tmp_path)
    dialog = _window(tmp_path, poni)
    try:
        dialog.frames = frames
        # the lane holds a real job, the calibrated frame's arm fit, until the window polls it on its timer
        dialog._start('predict', (dialog._calibrated_key(), dialog._seed_options()))
        dialog._timer.stop()
        assert dialog.active()
        dialog._render_frames()
        cells = _settle(dialog)
        assert dialog.active()  # the rings arrived while the lane is still busy
        assert cells == _expected(dialog, frames)
        dialog._timer.start()
        _wait_idle(dialog)  # the fit retires as before, and its prediction re-renders the rows
        assert dialog._seed_options()['prediction'] is not None
        assert _settle(dialog) == _expected(dialog, frames)
        # asked-for rings do not make the window busy: the lane starts a job meanwhile
        dialog.refine_calibrant.setCurrentText('CeO2')
        assert not dialog.active()
        dialog._start('predict', (dialog._calibrated_key(), dialog._seed_options()))
        assert dialog.active()
        _settle(dialog)
        _wait_idle(dialog)
        assert _settle(dialog) == _expected(dialog, frames)
        assert _close(dialog).status is PageCleanup.CLEAN
    finally:
        dialog.close()


def test_the_close_receipt_covers_the_ring_worker_and_nothing_arrives_after_it(tmp_path):
    frames, poni = _staged(tmp_path)
    dialog = _window(tmp_path, poni)
    try:
        dialog.frames = frames
        dialog._render_frames()  # asked for; nothing polled yet
        receipt = dialog.shutdown()
        assert receipt.status in (PageCleanup.CLEAN, PageCleanup.PENDING)  # PENDING while its worker returns
        assert _close(dialog).status is PageCleanup.CLEAN
        for _ in range(20):
            QtWidgets.QApplication.processEvents()
            time.sleep(.005)
        assert [text for text, _ in _cells(dialog)] == [PENDING] * len(frames)  # nothing handed over after close
        assert dialog.shutdown().status is PageCleanup.CLEAN
    finally:
        dialog.close()


def test_a_real_sessions_rings_are_predicted_off_the_gui_thread(tmp_path, request):
    """The maintainer's staged LaB6 session: five Eiger2 500k frames at del 13–53° and 11.27 keV under Int's Auto
    threshold, predicted from a typed rough geometry (the session's own calibrated frames do not fit an arm yet);
    a threshold edit then predicts all five again. The session's files are only read: the window writes to
    ``tmp_path``."""
    root = Path(os.environ.get('XDART_TEST_DATA', '/missing')) / 'calibration' / 'session-46e0f2e05eb4'
    if not (root / 'calibration-session.json').is_file():
        pytest.skip('XDART_TEST_DATA calibration session is not available')
    frames, arm, policy = stitch_calibration.load_calibration_session(root / 'calibration-session.json',
                                                                        with_arm=True, with_policy=True)
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path, detector='Eiger2_500k'))
    try:
        dialog.session_dir = tmp_path
        dialog._frame_motors = arm
        dialog._apply_fit_policy(policy)
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('manual'))
        for key, value in (('rough_distance', f"{policy['distance_seed_m'] * 1000:.12g}"), ('beam_pixel', '514, 256')):
            dialog._set_field_text(key, value)
        dialog.frames = frames
        dialog._render_frames()
        _settle(dialog)
        assert dialog._conditioning().mode == 'auto'
        with _KernelStarts() as starts:
            started = time.perf_counter()
            dialog.threshold_range._toggle[1].setChecked(False)  # a conditioning edit: every row is predicted again
            edit_s = time.perf_counter() - started
            first = _cells(dialog)
            settle_started = time.perf_counter()
            cells = _settle(dialog)
            settle_s = time.perf_counter() - settle_started
        request.node.user_properties.append(('edit_gui_s', f'{edit_s:.6f}'))
        request.node.user_properties.append(('rings_arrive_s', f'{settle_s:.6f}'))
        assert dialog._conditioning().mode == 'off'
        expected = _expected(dialog, frames)
        assert all(text not in ('—', 'none', PENDING) for text, _ in expected)
        assert starts.count('predict_frame_seed', gui=True) == 0  # parent polarity: five on the GUI thread
        assert [text for text, _ in first] == [PENDING] * len(frames)
        assert cells == expected
        assert starts.count('predict_frame_seed', gui=False) == len(frames)
        assert _close(dialog).status is PageCleanup.CLEAN
    finally:
        dialog.close()
