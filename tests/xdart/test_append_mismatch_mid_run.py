"""Append config mismatch reaching initialize_scan MID-RUN must stop the run
CLEANLY (typed error -> catch -> sigAppendMismatch -> user modal), never escape
imageThread.run() as an unhandled QThread exception.

The v1.0.1 beamline crash: a directory Int 1D Live/Append run was STOPPED, the
user switched to Int 2D and re-ran in Append over the existing 1D-config .nxs
targets.  ``initialize_scan``'s ``append_config_mismatch_check`` correctly
refused to mix configs into one append file, but its bare RuntimeError escaped
``run()`` (try/.../finally, NO except) as an unhandled exception -> the
_gui_main excepthook ("Unhandled exception in xdart GUI") and a dead run with
no user-facing explanation.  The Run-click mismatch modal (CF-2/CF-3) checks
only the single seed-derived target, so a per-file mismatch discovered during
the run lands on this in-thread guard — whose DELIVERY was the bug.

Production-wired on the seam being fixed: a REAL imageThread (full __init__,
real Qt signals, really ``start()``ed), a REAL 1D-config ``.nxs`` append
target on disk, a REAL tif frame read by fabio, and the REAL
initialize_scan -> process_scan -> run() path.  No fake sits on the
raise -> catch -> signal -> clean-stop seam.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from contextlib import contextmanager
from queue import Queue
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_HAS_PYQTGRAPH = importlib.util.find_spec("pyqtgraph") is not None
pytestmark = pytest.mark.skipif(
    not _HAS_PYQTGRAPH,
    reason="pyqtgraph GUI dependency is not installed",
)

if _HAS_PYQTGRAPH:
    import fabio
    import fabio.tifimage       # explicit: fabio does not eagerly load submodules
    import h5py
    from pyqtgraph import Qt
    from pyqtgraph.Qt import QtWidgets

    from xrd_tools.core.containers import PONI
    from xrd_tools.core.provenance import write_provenance
    from xrd_tools.session.readiness import AppendConfigMismatchError
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread,
    )
    from xdart.modules.live import LiveScan


STORED_1D_RUN_CONFIG = {
    # What the Int 1D run persisted into /entry/reduction of every target.
    "gi": False,
    "bai_1d_args": {"unit": "q_A^-1", "numpoints": 3000},
    "bai_2d_args": {"unit": "q_A^-1", "npt_rad": 500, "npt_azim": 500},
}

CURRENT_2D_ARGS = {
    # The re-run's Int 2D settings: same Standard mode, different 2D grid —
    # the case where the old reason read "processed: Standard · current:
    # Standard" and named nothing.
    "bai_1d_args": {"unit": "q_A^-1", "numpoints": 3000},
    "bai_2d_args": {"unit": "q_A^-1", "npt_rad": 1000, "npt_azim": 500},
}


def _qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _write_1d_config_target(path, labels):
    """A minimal integrated v2 target carrying the 1D run's stored config."""
    labels = np.asarray(labels, dtype=np.int64)
    q = np.linspace(0.1, 1.0, 4, dtype=np.float32)
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        g1 = entry.create_group("integrated_1d")
        g1.attrs["NX_class"] = "NXdata"
        g1.attrs["signal"] = "intensity"
        g1.attrs["axes"] = ["frame_index", "q"]
        g1.create_dataset("frame_index", data=labels)
        q_ds = g1.create_dataset("q", data=q)
        q_ds.attrs["units"] = "q_A^-1"
        g1.create_dataset(
            "intensity",
            data=np.arange(labels.size * q.size, dtype=np.float32).reshape(
                labels.size, q.size
            ),
        )
        write_provenance(h5, config=STORED_1D_RUN_CONFIG, host="")


def _real_append_thread(tmp_path):
    """A REAL imageThread configured like the beamline re-run: Append over an
    existing 1D-config target, current config differing in the 2D leg."""
    out = tmp_path / "out"
    out.mkdir()
    target = out / "scan.nxs"
    # Frame 4 is NOT in the stored labels, so the run-start append-skip
    # snapshot does not skip it and the run reaches initialize_scan.
    _write_1d_config_target(target, [1, 2, 3])
    raw = tmp_path / "scan_0004.tif"
    fabio.tifimage.TifImage(data=np.ones((4, 4), dtype=np.uint16)).write(
        str(raw))

    current_scan = LiveScan(
        "scan",
        data_file=str(target),
        static=True,
        bai_1d_args=dict(CURRENT_2D_ARGS["bai_1d_args"]),
        bai_2d_args=dict(CURRENT_2D_ARGS["bai_2d_args"]),
    )

    thread = imageThread(
        Queue(),                     # command_queue
        {},                          # scan_args
        threading.RLock(),           # file_lock
        "",                          # fname
        str(out),                    # h5_dir
        "scan",                      # scan_name
        False,                       # single_img
        PONI(dist=0.2, poni1=0.1, poni2=0.1, wavelength=1e-10),
        "Image Series",              # inp_type
        str(raw),                    # img_file
        str(tmp_path),               # img_dir
        False,                       # include_subdir
        "tif",                       # img_ext
        False,                       # series_average
        None,                        # meta_ext
        "",                          # file_filter
        None,                        # mask_file
        "Append",                    # write_mode
        "None",                      # bg_type
        "",                          # bg_file
        "",                          # bg_dir
        None,                        # bg_matching_par
        "",                          # bg_match_fname
        "",                          # bg_file_filter
        1.0,                         # bg_scale
        None,                        # bg_norm_channel
        False,                       # gi
        None,                        # th_mtr
        1,                           # sample_orientation
        0.0,                         # tilt_angle
        "q_total",                   # gi_mode_1d
        "qip_qoop",                  # gi_mode_2d
        "start",                     # command
        current_scan,                # scan
        live_mode=False,
        max_cores=1,
    )
    return thread, target


@contextmanager
def _capture_unhandled():
    """Record anything reaching sys.excepthook / threading.excepthook — the
    exact no-unhandled-exception contract run() must uphold."""
    seen = []
    old_sys = sys.excepthook
    old_threading = threading.excepthook
    sys.excepthook = lambda *args: seen.append(args)
    threading.excepthook = lambda args: seen.append(args)
    try:
        yield seen
    finally:
        sys.excepthook = old_sys
        threading.excepthook = old_threading


def test_real_thread_append_mismatch_stops_cleanly_and_signals(tmp_path):
    """The whole delivery, end to end on the REAL QThread: the typed error
    raised by the REAL initialize_scan is caught inside the run, the run stops
    like a user Stop, sigAppendMismatch carries the user-facing message, the
    append target is byte-identical — and NOTHING reaches the excepthook."""
    _qapp()
    thread, target = _real_append_thread(tmp_path)
    before = target.read_bytes()

    messages = []
    thread.sigAppendMismatch.connect(
        messages.append, Qt.QtCore.Qt.ConnectionType.DirectConnection)

    with _capture_unhandled() as unhandled:
        thread.start()
        assert thread.wait(30000), "wrangler thread did not finish"

    assert unhandled == []                      # never the _gui_main excepthook
    assert thread.command == "stop"             # the user-Stop clean-stop path
    assert len(messages) == 1
    message = messages[0]
    # The message names WHAT changed and that the target survived.
    assert "Integration settings changed mid-run" in message
    assert "2D radial points" in message
    assert "scan.nxs was preserved" in message
    assert "Replace" in message
    assert target.read_bytes() == before        # guard preserved the target


def test_live_watch_append_mismatch_stops_cleanly(tmp_path):
    """The live-watch call site (Phase 3): a NEW scan appearing while watching
    hits initialize_scan; the typed error must stop the watch cleanly AND the
    shared end-of-run tail must still force-flush the PREVIOUS scan (exactly
    like a user Stop while watching).  Drives the REAL process_scan and the
    REAL _handle_append_config_mismatch with the REAL typed error."""
    saves = []
    scan_a = SimpleNamespace(
        name="a",
        data_file=str(tmp_path / "a.nxs"),
        frames=SimpleNamespace(index=[]),
        skip_2d=False,
        _save_to_nexus=lambda: saves.append("a"),
    )

    queue = [
        ("/tmp/a_0001.tif", "a", 1, np.ones((2, 2)), {}),
        (None, None, None, None, None),          # collect glob exhausted
        ("/tmp/b_0001.tif", "b", 1, np.ones((2, 2)), {}),
    ]

    def next_image():
        if queue:
            return queue.pop(0)
        return None, None, None, None, None

    check = SimpleNamespace(
        ok=False, mismatched_fields=("2D radial points",),
        processed_label="Standard", current_label="Standard", reason="")

    init_calls = []

    def initialize_scan():
        init_calls.append(len(init_calls))
        if len(init_calls) == 1:
            return scan_a
        raise AppendConfigMismatchError(
            "Integration settings changed mid-run (2D radial points); the "
            "append target b.nxs was preserved. Switch write mode to "
            "Replace, or revert settings, to continue.",
            check,
        )

    emitted = []
    dispatched = []
    host = SimpleNamespace(
        command="start",
        batch_mode=False,
        live_mode=True,
        single_img=False,
        xye_only=False,
        img_file="/tmp/a_0001.tif",
        poni=None,
        scan_name="a",
        file_lock=threading.RLock(),
        _frames_since_save=0,
        _active_scan=None,
        _perf=None,
        _live_execution=lambda: "serial",
        showLabel=SimpleNamespace(emit=lambda *_: None),
        sigUpdate=SimpleNamespace(emit=lambda *_: None),
        sigAppendMismatch=SimpleNamespace(emit=emitted.append),
        _wait_if_paused=lambda: None,
        get_next_image=next_image,
        _middle_truncate=lambda text, max_len=40: text,
        initialize_scan=initialize_scan,
        get_background=lambda *_: 0.0,
        _flush_xye_buffer=lambda *_args, **_kw: None,
        _save_due=lambda scan, force=False: force,
        _prime_append_skip_snapshots_for_run=lambda: None,
        _dispatch_batch=lambda scan, pending, force_save=False: dispatched.append(
            tuple(item[1] for item in pending)) or len(pending),
        _process_one=lambda *a, **k: None,
    )

    @contextmanager
    def _noop_bracket(_scan):
        yield

    host._h5pool_bracket = _noop_bracket
    host.flush_serial_tail = MethodType(imageThread.flush_serial_tail, host)
    host._handle_append_config_mismatch = MethodType(
        imageThread._handle_append_config_mismatch, host)

    MethodType(imageThread.process_scan, host)()    # must NOT raise

    assert init_calls == [0, 1]                 # scan "a", then the mismatch
    assert host.command == "stop"
    assert len(emitted) == 1
    assert "b.nxs was preserved" in emitted[0]
    assert dispatched == [(1,)]                 # scan "a" frame was processed
    assert saves == ["a"]                       # previous scan tail flushed


def test_wrangler_widget_surfaces_one_warning_modal_and_status():
    """The GUI side of the delivery: sigAppendMismatch -> _on_append_mismatch
    shows ONE single-button warning modal (the Run-click mismatch modal's
    presentation, informational because the run already stopped) and leaves
    the stop reason in the status text after dismissal."""
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler as iw_mod
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import (
        imageWrangler,
    )

    boxes = []

    class _FakeMessageBox:
        Warning = object()

        def __init__(self, parent=None):
            self.parent = parent
            self.execs = 0
            boxes.append(self)

        def setIcon(self, icon):
            self.icon = icon

        def setWindowTitle(self, title):
            self.title = title

        def setText(self, text):
            self.text = text

        def exec(self):
            self.execs += 1
            return 0

    statuses = []
    host = SimpleNamespace(_set_status_text=statuses.append)
    message = (
        "Integration settings changed mid-run (2D radial points); the append "
        "target scan.nxs was preserved. Switch write mode to Replace, or "
        "revert settings, to continue."
    )

    original = iw_mod.QMessageBox
    iw_mod.QMessageBox = _FakeMessageBox
    try:
        imageWrangler._on_append_mismatch(host, message)
    finally:
        iw_mod.QMessageBox = original

    assert len(boxes) == 1
    box = boxes[0]
    assert box.icon is _FakeMessageBox.Warning
    assert box.title == "Integration settings changed mid-run"
    assert box.text == message
    assert box.execs == 1
    assert statuses[-1] == message              # reason survives the modal
