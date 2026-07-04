# -*- coding: utf-8 -*-
"""
@author: walroth
"""

# Standard library imports
import logging
import os
from queue import Queue
import threading
import traceback

# Other imports
import numpy as np
from pathlib import Path

# Qt imports
from pyqtgraph import Qt
from pyqtgraph.parametertree import Parameter

# This module imports
from xdart.utils.h5pool import get_pool as _get_h5pool
from xrd_tools.io.export import write_xye
from .qt_nexus_sink import _is_append_axis_mismatch

logger = logging.getLogger(__name__)


# MEM-1a: the wrangler→GUI live display hand-off (``_published_frames``) stashes
# one fully-hydrated LiveFrame per frame for the GUI's ``update_data`` to pop.
# In live mode each retained frame still holds its ~18 MB raw (upcast ~64 MB
# float64) until it leaves the write-side staging window, so if the GUI thread
# falls behind the producer an *unbounded* dict grows to tens of GB → OOM.
# Bounding it with DROP-OLDEST is safe post-8a: the frame is already durable via
# the sink write, and the display re-hydrates any evicted label on demand (the
# store-first path).  The dropped entry is always the OLDEST undrained frame —
# never the freshly-signalled idx the GUI is about to consume next.
_PUBLISHED_FRAMES_CAP = 128


class _BoundedFrameHandoff(dict):
    """Insertion-ordered dict capped at ``cap`` entries (drop-oldest on insert).

    A plain ``dict`` with a size guard: every ``__setitem__`` that pushes past
    ``cap`` evicts the oldest key(s).  ``pop``/``get``/``clear`` are inherited
    unchanged, so the existing consumer (``update_data``: ``pop(idx, None)``,
    ``get(idx)``) keeps working — a dropped idx simply reads back as ``None``,
    which the consumer already tolerates.
    """

    def __init__(self, *args, cap: int = _PUBLISHED_FRAMES_CAP, **kwargs):
        super().__init__(*args, **kwargs)
        self._cap = max(1, int(cap))
        self._drop_warned = False

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        over = len(self) - self._cap
        if over > 0:
            for stale in list(self.keys())[:over]:
                super().pop(stale, None)
            if not self._drop_warned:
                self._drop_warned = True
                logger.warning(
                    "GUI behind producer: dropping oldest display hand-offs "
                    "(cap=%d); frames remain on disk and re-hydrate on demand",
                    self._cap)

    def clear(self):
        super().clear()
        self._drop_warned = False


# Sentinel used by ``_apply_threshold_inline`` to mark out-of-band
# pixels.  pyFAI's CSR integrator auto-skips NaN at integrate time
# without invalidating the mask CRC, so the per-frame threshold
# filter survives without forcing a per-frame LUT rebuild.
_THRESHOLD_NAN = np.float32(np.nan)

# Default cadence: flush scan state to disk every N frames in batch
# / live modes.  Subclasses can override per-instance via the
# ``LIVE_SAVE_INTERVAL`` attribute if they want a different rhythm.
_LIVE_SAVE_INTERVAL = 8
# 1D-only .nxs writes are small and cheap, so flush far less often: it is the
# fixed per-save overhead (not per-frame compute) that made long Int-1D scans
# crawl as the frame count grew.  2D keeps the tight default so peak RAM stays
# bounded.  (PERF-2)
# This is now an UPPER bound on save spacing, not the effective cadence: the
# persist-before-evict fix (LiveFrameSeries._persisted + mark_persisted, and the
# _save_due cap bound in imageWranglerThread) guarantees a save fires before the
# unsaved in-memory set reaches _in_memory_cap, so no frame's int_1d is ever
# evicted before it's written — the high interval is safe on scans longer than
# the cap.  Effective cadence is therefore min(this, cap-margin).
_LIVE_SAVE_INTERVAL_1D = 1000


class _CommandCancelToken:
    """Duck-typed ssrl CancelToken bound to a wrangler thread command."""

    def __init__(self, owner):
        self._owner = owner

    @property
    def cancelled(self):
        return getattr(self._owner, 'command', None) == 'stop'


class wranglerWidget(Qt.QtWidgets.QWidget):
    """Base class for wranglers. Extending this ensures all methods,
    signals, and attributes expected by ttheta_widget are present.
    Threads should be started by use of sigStart.emit, which ensures
    tthetaWidget handles initiation.
    
    attributes:
        command_queue: Queue, used to send commands to thread
        file_lock, mp.Condition, process safe lock for file access
        fname: str, path to data file
        parameters: pyqtgraph Parameter, stores parameters from user
        scan_name: str, current scan name, used to handle syncing data
        scan_args: dict, used as **kwargs in scan initialization.
            see LiveScan.
        thread: wranglerThread or subclass, QThread for controlling
            processes
    
    methods:
        enabled: Enables or disables interactivity
        set_fname: Method to safely change file name
        setup: Syncs thread parameters prior to starting
    
    signals:
        finished: Should be connected to thread.finished signal
        sigStart: Tells tthetaWidget to start the thread and prepare
            for new data.
        sigUpdateData: int, signals a new frame has been added.
        sigUpdateFile: (str, str, bool, str, bool, bool), sends new scan_name, file name
            GI flag (grazing incidence), theta motor for GI, single_image and
            series_average flag to static_scan_Widget.
    """
    sigStart = Qt.QtCore.Signal()
    sigUpdateData = Qt.QtCore.Signal(int)
    # sigUpdateFrame = Qt.QtCore.Signal(dict)
    sigUpdateFile = Qt.QtCore.Signal(str, str, bool, str, bool, bool)
    sigUpdateGI = Qt.QtCore.Signal(bool)
    # GI move (Stage B): hands the available SPEC incidence-motor columns to the
    # integrator panel's GI motor dropdown (the integrator owns the selection).
    sigGIMotorOptions = Qt.QtCore.Signal(list)
    finished = Qt.QtCore.Signal()
    started = Qt.QtCore.Signal()
    # Pause/Resume (Phase B): sigPaused fires once the run is frozen at a frame
    # boundary (the host then LIFTS the freeze guard for browsing); sigResuming
    # fires just before resuming (the host RE-ENGAGES the guard FIRST).  Emitted
    # only by wranglers that support pause (image wrangler); harmless elsewhere.
    sigPaused = Qt.QtCore.Signal()
    sigResuming = Qt.QtCore.Signal()

    def __init__(self, fname, file_lock, parent=None):
        """fname: str, file path
        file_lock: mp.Condition, process safe lock
        """
        super().__init__(parent)
        self.file_lock = file_lock
        self.fname = fname
        self.scan_name = 'null_thread'
        self.parameters = Parameter.create(
            name='wrangler_widget', type='int', value=0
        )
        self.scan_args = {}

        self.command_queue = Queue()
        self.thread = wranglerThread(self.command_queue, self.scan_args, self.fname, self.file_lock, self)
        self.thread.finished.connect(self.finished.emit)
        self.thread.started.connect(self.started.emit)
        self.thread.sigUpdate.connect(self.sigUpdateData.emit)
        # self.thread.sigUpdateFrame.connect(self.sigUpdateFrame.emit)
        self.thread.sigUpdateGI.connect(self.sigUpdateGI.emit)

        # Shared run-controls (CONTROLS section): the staticWidget owns one
        # StaticControls widget and ATTACHES it to the active wrangler, which
        # aliases its own control refs onto the shared widgets so all existing
        # run-lifecycle logic drives them.  None until attached.
        self._controls = None
        self._control_conns = []

    def enabled(self, enable):
        """Use this function to control what is enabled and disabled
        during integration.
        """
        pass

    # ── Shared run-controls (CONTROLS section) ──────────────────────────
    def controls_profile(self):
        """Per-wrangler capability descriptor for the shared StaticControls:
        the mode items to show + whether Live / Batch / cores apply.  Base
        default = no Live/Batch (subclasses override)."""
        return {'modes': None, 'live': False, 'batch': False, 'cores': True}

    def attach_controls(self, controls):
        """Adopt the shared StaticControls widget.  Base just stores the ref;
        subclasses override to alias their own control attributes onto the shared
        widgets and wire the shared signals to their handlers (tracking
        connections via _connect_control for detach_controls)."""
        self._controls = controls

    def _connect_control(self, signal, slot):
        """Connect a shared-control signal to one of THIS wrangler's handlers and
        record it, so detach_controls (on wrangler swap) disconnects exactly
        these — preventing a stale wrangler from double-dispatching a click."""
        signal.connect(slot)
        self._control_conns.append((signal, slot))

    def detach_controls(self):
        """Disconnect this wrangler's shared-control connections (on swap, before
        the next wrangler attaches)."""
        for signal, slot in self._control_conns:
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._control_conns = []

    # ── Group-header toggles (UI-1, #81) ────────────────────────────────
    # Maps a toggle-group's name (e.g. 'GI') to its hidden enabling bool
    # child ('Grazing').  _install_group_toggles puts a REAL checkbox on
    # the group's header row: the checkbox is the on/off control, driving
    # the hidden bool that stays the source of truth the wrangler reads
    # (hidden so it can't repaint-uncheck while the tree is disabled
    # mid-run, #56).  Checking expands the group, unchecking collapses it;
    # a manual chevron expand just peeks at the options — it does NOT
    # enable the feature.
    _GROUP_TOGGLES = {}

    @staticmethod
    def _toggle_check_state(on):
        return (Qt.QtCore.Qt.CheckState.Checked if on
                else Qt.QtCore.Qt.CheckState.Unchecked)

    def _install_group_toggles(self, tree):
        """Add a checkbox to each _GROUP_TOGGLES group's header item and wire
        it both ways to the group's hidden enabling bool.  Call once, after
        ``tree.setParameters`` (the header items must exist)."""
        self._group_toggle_items = []
        for grp_name, bool_name in self._GROUP_TOGGLES.items():
            try:
                grp = self.parameters.child(grp_name)
                bool_param = grp.child(bool_name)
                item = next(iter(grp.items))
            except Exception:
                logger.debug("group-toggle install skipped for %s", grp_name,
                             exc_info=True)
                continue
            item.setFlags(item.flags()
                          | Qt.QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, self._toggle_check_state(bool_param.value()))
            self._group_toggle_items.append((item, grp, bool_param))
            bool_param.sigValueChanged.connect(self._sync_group_toggle_from_bool)
            # pyqtgraph's ParameterItem.optsChanged ends with updateFlags(),
            # which rebuilds the header's flags from the param opts and drops
            # ItemIsUserCheckable (any setOpts — expanded, visible — strips
            # the checkbox).  Re-assert it after every opts change; connected
            # AFTER the item's own optsChanged so it runs post-updateFlags.
            grp.sigOptionsChanged.connect(self._reassert_group_toggle_flags)
        if self._group_toggle_items:
            tree.itemChanged.connect(self._on_group_toggle_item_changed)

    def _reassert_group_toggle_flags(self, _param=None, _opts=None):
        checkable = Qt.QtCore.Qt.ItemFlag.ItemIsUserCheckable
        for item, _grp, _bool_param in getattr(self, '_group_toggle_items', ()):
            if not (item.flags() & checkable):
                item.setFlags(item.flags() | checkable)

    def _on_group_toggle_item_changed(self, item, column):
        """User (un)checked a toggle-group header: drive the hidden bool and
        open/fold the group to match."""
        if column != 0:
            return
        for it, grp, bool_param in getattr(self, '_group_toggle_items', ()):
            if it is item:
                on = (item.checkState(0)
                      == Qt.QtCore.Qt.CheckState.Checked)
                if bool(bool_param.value()) != on:
                    bool_param.setValue(on)
                    save = getattr(self, '_save_to_session', None)
                    if save is not None:
                        save()
                grp.setOpts(expanded=on)
                return

    def _sync_group_toggle_from_bool(self, param, value):
        """Programmatic bool change (session restore etc.): reflect it into
        the header checkbox."""
        for item, grp, bool_param in getattr(self, '_group_toggle_items', ()):
            if bool_param is param:
                state = self._toggle_check_state(bool(value))
                if item.checkState(0) != state:
                    item.setCheckState(0, state)
                return

    # ── Status label (specLabel / statusLabel) ──────────────────────────
    # A plain QLabel's minimum width is its full text width, so a long
    # status message (e.g. the live-GI clip advisory, ~180 chars) forces
    # the WHOLE window to expand horizontally.  Subclasses must route
    # status text through _set_status_text and call _guard_status_label
    # once after building their UI.

    def _status_label(self):
        """The status QLabel that messages route to.  Prefers the shared
        control-layer ``statusLabel`` (StaticControls) when controls are
        attached, so the message bar lives in ONE place (the control stack)
        instead of the wrangler's own orphaned label; falls back to the
        subclass's own label / specUI specLabel when standalone."""
        controls = getattr(self, '_controls', None)
        cl = getattr(controls, 'statusLabel', None) if controls is not None else None
        if cl is not None:
            return cl
        label = getattr(self, 'statusLabel', None)
        if label is not None:
            return label
        ui = getattr(self, 'ui', None)
        return getattr(ui, 'specLabel', None) if ui is not None else None

    def _guard_status_label(self):
        """Stop the status label from driving the window's minimum width:
        with an Ignored horizontal policy the label takes whatever width the
        layout gives it and overlong text clips instead of growing the window."""
        label = self._status_label()
        if label is None:
            return
        policy = label.sizePolicy()
        policy.setHorizontalPolicy(Qt.QtWidgets.QSizePolicy.Policy.Ignored)
        label.setSizePolicy(policy)

    def _status_bar(self):
        """The main window's BOTTOM QStatusBar, when this widget is hosted in a
        QMainWindow (the normal app).  None when standalone (tests / popped-out
        wranglers) — status then falls back to the elide-safe label."""
        try:
            win = self.window()
            fn = getattr(win, 'statusBar', None)
            return fn() if callable(fn) else None
        except Exception:
            return None

    def _set_status_text(self, text):
        """Route run/browse status to the main window's bottom status bar (the
        message bar).  Falls back to the elide-safe status label when there is no
        status bar (standalone): elides to the label's width, full text in the
        tooltip — see _guard_status_label."""
        text = text or ''
        bar = self._status_bar()
        if bar is not None:
            bar.showMessage(text)
            return
        label = self._status_label()
        if label is None:
            return
        label.setToolTip(text)
        if label.isVisible() and label.width() > 0:
            metrics = Qt.QtGui.QFontMetrics(label.font())
            text = metrics.elidedText(
                text, Qt.QtCore.Qt.TextElideMode.ElideRight, label.width() - 4)
        label.setText(text)

    def setup(self):
        """Sets the thread child object. Called by tthetaWidget prior
        to starting thread.
        """
        # Disconnect old thread signals to avoid duplicate emissions
        try:
            self.thread.finished.disconnect(self.finished.emit)
            self.thread.started.disconnect(self.started.emit)
            self.thread.sigUpdate.disconnect(self.sigUpdateData.emit)
            self.thread.sigUpdateGI.disconnect(self.sigUpdateGI.emit)
        except (TypeError, RuntimeError):
            pass  # Signals were never connected or already disconnected
        self.thread = wranglerThread(self.command_queue, self.scan_args, self.fname, self.file_lock, self)
        self.thread.finished.connect(self.finished.emit)
        self.thread.started.connect(self.started.emit)
        self.thread.sigUpdate.connect(self.sigUpdateData.emit)
        self.thread.sigUpdateGI.connect(self.sigUpdateGI.emit)

    def set_fname(self, fname):
        """Changes fname attribute of self and thread.
        args:
            fname: str, path for new file.
        """
        with self.file_lock:
            if not self.thread.isRunning():
                self.fname = fname
                self.thread.fname = fname


class wranglerThread(Qt.QtCore.QThread):
    """Base class for wranglerThreads. Used to manage processes
    including data and command queues. Subclasses should override the
    run method.
    
    attributes:
        command_q: mp.Queue, queue to send commands to process
        file_lock: mp.Condition, process safe lock for file access
        fname: str, path to data file.
        input_q: mp.Queue, queue for commands sent from parent
        signal_q: mp.Queue, queue for commands sent from process
        scan_args: dict, used as **kwargs in scan initialization.
            see LiveScan.
    
    methods:
        run: Called by start, main thread task.
    
    signals:
        sigUpdate: int, signals a new frame has been added.
        sigUpdateFile: (str, str, bool, str, bool, bool), sends new scan_name, file name
            GI flag (grazing incidence), theta motor for GI, single_image and
            series_average flag to static_scan_Widget.
        sigUpdateGI: bool, signals the grazing incidence condition has changed.
    """
    sigUpdate = Qt.QtCore.Signal(int)
    # sigUpdateFrame = Qt.QtCore.Signal(dict)
    sigUpdateFile = Qt.QtCore.Signal(str, str, bool, str, bool, bool)
    sigUpdateGI = Qt.QtCore.Signal(bool)

    # Save cadence (frames between disk flushes), mode-aware: a 1D-only run
    # (``scan.skip_2d``) flushes every ``_LIVE_SAVE_INTERVAL_1D`` frames; a 2D
    # run keeps the tight ``_LIVE_SAVE_INTERVAL`` for bounded RAM.  An instance
    # may still pin a value (e.g. a test) by assigning ``LIVE_SAVE_INTERVAL``.
    @property
    def LIVE_SAVE_INTERVAL(self) -> int:
        override = getattr(self, "_live_save_interval_override", None)
        if override is not None:
            return int(override)
        if getattr(getattr(self, "scan", None), "skip_2d", False):
            return _LIVE_SAVE_INTERVAL_1D
        return _LIVE_SAVE_INTERVAL

    @LIVE_SAVE_INTERVAL.setter
    def LIVE_SAVE_INTERVAL(self, value: int) -> None:
        self._live_save_interval_override = int(value)

    def __init__(self, command_queue, scan_args, fname, file_lock,
                 parent=None):
        """command_queue: mp.Queue, queue for commands sent from parent
        scan_args: dict, used as **kwargs in scan initialization.
            see LiveScan.
        fname: str, path to data file.
        file_lock: mp.Condition, process safe lock for file access
        """
        super().__init__(parent)
        self.input_q = command_queue # thread queue
        self.scan_args = scan_args
        self.fname = fname
        self.file_lock = file_lock
        self.signal_q = Queue()
        self.command_q = Queue()
        # RS-2: serializes command TRANSITIONS between the GUI (pause/resume
        # check-then-set) and the worker's self-stop writes (write-failure
        # stop, GI freeze abort) — without it a self-stop landing between the
        # GUI's check and its 'pause' write was silently revived.
        self.command_lock = threading.Lock()

        # ── Shared batch-engine state ────────────────────────────────
        # Subclasses can override any of these before .start() (or
        # via their own __init__) — the defaults are the "no batch
        # features active" zero state.

        # XYE write buffer + lock.  Populated during integration in
        # workers; drained at end of batch by _flush_xye_buffer.
        self._xye_buffer: list = []
        self._xye_lock = threading.Lock()

        # Per-batch save cadence counter.  Wraps to zero each time
        # _save_to_disk fires.
        self._frames_since_save = 0

        # In-memory hand-off of just-integrated frames to the main
        # thread so it doesn't have to round-trip through disk.  The
        # main thread's update_data consumes this dict.
        self._published_frames: dict = _BoundedFrameHandoff()

        # Threshold filtering — subclass sets these from its UI; the
        # base default is "no threshold" so nexus / other wranglers
        # that don't expose a threshold UI pay nothing.
        self.apply_threshold = False
        self.threshold_min = 0
        self.threshold_max = 0
        # Auto-mask the uint16 ceiling (65535) as a saturated/dead sentinel.
        # ON by default = the long-standing behaviour; wranglers that expose
        # the Intensity-Threshold UI override this from the param tree.
        self.mask_sentinel = True

        # Sub-label appended to log lines (e.g. "[Subtracted bg.tif]"
        # for SPEC bg-subtraction mode).  Empty string = no append.
        self.sub_label = ''

        # Mode flags read by the dispatch loops + the GUI's
        # wrangler_finished handler.
        self.batch_mode = False
        self.xye_only = False
        self.max_cores = 1
        self._reduction_session = None
        self._reduction_session_key = None
        # Streaming (PERF-4b) session + its QtNexusSink, kept on dedicated slots
        # because one persistent session spans the WHOLE scan (the chunked cache
        # keys on per-chunk n_workers, which varies).  Finished at scan end by
        # _close_reduction_session.
        self._streaming_session = None
        self._streaming_sink = None
        self._streaming_scan_id = None
        self._streaming_record_store = None
        # 4c-1/4d: the streaming register/submit/pause seam (created in
        # _get_streaming_session, dies in _close_reduction_session).  Initialised
        # here so the `scan_session` property + GUI run-state reads never hit an
        # AttributeError before the first streaming session opens.
        self._scan_session_adapter = None
        # BLOCKER 1: id of the scan whose whole-scan GI grid pre-pass has run, so
        # the freeze happens once per scan (not per chunk).  Reset on scan close.
        self._gi_prepass_scan_id = None
        # Set by _close_reduction_session when a streaming write/sink failure
        # surfaces from finish() — so the run can't report a false "success".
        self._reduction_write_error = None

    def run(self):
        """Main task. Subclasses (e.g. imageThread) override this."""
        pass

    # ── Shared batch helpers ────────────────────────────────────────

    def _cancel_token(self):
        return _CommandCancelToken(self)

    def _get_reduction_session(self, key, factory):
        """Return the persistent headless reduction session for *key*.

        The session owns the executor and per-thread pyFAI integrators for the
        scan/run lifetime.  The caller supplies a key that includes scan identity
        and execution policy; changing either closes the old session and opens a
        fresh one from the provided factory.
        """
        if self._reduction_session is not None and self._reduction_session_key == key:
            return self._reduction_session

        self._close_reduction_session()
        self._reduction_session = factory()
        self._reduction_session_key = key
        return self._reduction_session

    def _reduction_session_key_for(self, scan, plan, n_workers):
        try:
            n_workers = int(n_workers or 1)
        except (TypeError, ValueError):
            n_workers = 1
        return (
            id(scan),
            str(getattr(scan, "name", "scan")),
            str(getattr(scan, "data_file", "")),
            max(1, n_workers),
            bool(getattr(scan, "gi", False)),
            bool(getattr(scan, "skip_2d", False)),
            id(plan),
        )

    def _close_reduction_session(self):
        session = self._reduction_session
        self._reduction_session = None
        self._reduction_session_key = None
        # The streaming session's finish() drains the writer thread + does the
        # final QtNexusSink flush (save + XYE + end-of-run signal), so closing
        # it here is the streaming batch's end-of-scan write.
        streaming = self._streaming_session
        self._streaming_session = None
        self._streaming_sink = None
        self._streaming_scan_id = None
        self._streaming_record_store = None
        self._scan_session_adapter = None        # 4c-1: adapter dies with the session
        self._gi_prepass_scan_id = None      # next scan re-runs its own pre-pass
        # BLOCKER 2: finish() is fail-loud — a streaming sink/write failure now
        # RAISES instead of being silently swallowed (the user must not think a
        # failed write succeeded).  Close BOTH sessions even if the first raises
        # (wrap each individually + collect), then surface the failure loudly.
        errors = []

        def _submitted_count(sess):
            value = getattr(sess, "frames_submitted", None)
            if value is None:
                return None
            try:
                return int(value() if callable(value) else value)
            except Exception:
                return None

        def _report_result(sess, res):
            if res is None:
                return
            submitted = _submitted_count(sess)
            try:
                written = int(getattr(res, "n_processed"))
            except Exception:
                written = None
            if bool(getattr(res, "cancelled", False)) and written is not None:
                logger.info("Total Files Processed (durable after cancel): %d",
                            written)
            if submitted is None or written is None or submitted == written:
                return
            unwritten = max(0, submitted - written)
            msg = (
                f"Stopped with {unwritten} frame(s) un-written "
                f"(submitted={submitted}, written={written}) — source data "
                "intact; re-run Append/batch to recover"
            )
            logger.warning(msg)
            show = getattr(self, "showLabel", None)
            if show is not None:
                try:
                    show.emit(msg)
                except Exception:
                    pass

        for sess in (session, streaming):
            if sess is not None:
                try:
                    # #4 (codex): bound the writer-thread join so a stalled
                    # NFS/pyFAI worker can't wedge Stop/close indefinitely.
                    # 60 s is a generous ceiling for beamline conditions.
                    res = sess.finish(join_timeout=60.0)
                    _report_result(sess, res)
                except Exception as exc:
                    errors.append(exc)
                    if _is_append_axis_mismatch(exc):
                        logger.debug(
                            "append mismatch already reported by sink abort; "
                            "suppressing duplicate traceback",
                            exc_info=True,
                        )
                    else:
                        logger.error(
                            "reduction session WRITE FAILED on close: %s",
                            exc,
                            exc_info=True,
                        )
        if errors:
            self._reduction_write_error = errors[0]
            msg = (f"Save FAILED — output .nxs may be incomplete: {errors[0]}")
            show = getattr(self, "showLabel", None)
            if show is not None:
                try:
                    show.emit(msg)
                except Exception:
                    pass
            # A failed write is serious — stop the run rather than process
            # further scans onto a broken output.  Under command_lock so a
            # concurrent GUI pause() can't overwrite this stop (RS-2).
            # getattr: tests drive this on duck holders without the lock.
            if getattr(self, "command", None) is not None:
                _lock = getattr(self, "command_lock", None)
                if _lock is not None:
                    with _lock:
                        self.command = 'stop'
                else:
                    self.command = 'stop'

    @property
    def scan_session(self):
        """The active streaming session seam (``ScanSessionAdapter``) or None.

        4d: the single read-only accessor the GUI consults for run-state
        (``is_running`` / ``is_paused``) instead of poking the private adapter
        slot, and the seam 4f's public ``xrd_tools.session.ScanSession`` bridge
        hangs off.  None when no streaming session is open (true-live watch and
        the reintegrate-via-integratorThread path have no adapter — callers fall
        back to their own run-state cache)."""
        return self._scan_session_adapter

    def _resolve_frame_mask(self, scan, img_data):
        """Return a stable per-scan "bad pixel" mask cached on the scan.

        Computed once from ``img_data < 0`` of the first frame seen
        by this scan; reused for every subsequent frame.  Keeping
        the mask stable across frames is what lets pyFAI's CSR
        engine cache stay valid — a single pixel changing in the
        mask invalidates the cache and forces a ~250 ms LUT rebuild
        (observed on Eiger scans where saturation flicker shifts the
        mask CRC frame-to-frame).

        Per-frame threshold filtering is NOT routed through this
        mask — see :meth:`_apply_threshold_inline` for that path
        (NaN-sentinel in the data, mask CRC unchanged).

        F3: callers in the parallel section should pre-warm the
        cache via :meth:`_prewarm_frame_mask` on the main thread
        BEFORE submitting work, so the cache is fully populated
        when N workers read it.  Without the prewarm, the first N
        workers all see ``None`` and race to write the same value —
        currently safe because every worker computes the SAME mask,
        but the invariant isn't enforced by the code and a future
        change (e.g. per-worker thresholding) could break it.
        """
        cached = getattr(scan, '_cached_data_mask', None)
        if cached is None:
            sat_fired = 0           # R3-A: how many saturation pixels were cut
            frame_size = 0
            try:
                from xdart.modules.reduction import compute_bad_pixel_mask
                from ..display_logic import integer_saturation_ceiling
                from xrd_tools.core.invalid import saturation_pixels
                arr0 = np.asarray(img_data)
                frame_size = int(arr0.size)
                mask_sat = bool(getattr(self, 'mask_sentinel', True))
                # ONE masking implementation (xdart.modules.reduction) shared
                # with the reintegrate path so live ≡ reintegrate on the same
                # frame.  Pass the DISPLAY policy's ceiling (its legacy 65535
                # float fallback) so a float-typed raw masks identically on both
                # paths — keeping the live≡batch≡reload equivalence spine safe.
                # "Mask Saturated" (mask_sentinel) is the AUTHORITATIVE on/off:
                # OFF -> compute_bad_pixel_mask returns None -> nothing masked
                # (strong Bragg peaks that saturate are KEPT); ON -> negatives +
                # uint32 sentinel + fraction-guarded ceiling.  Computed-once +
                # cached, so pyFAI's CSR mask-CRC stays stable frame-to-frame.
                ceil = integer_saturation_ceiling(arr0)
                idx = compute_bad_pixel_mask(
                    arr0, mask_saturation=mask_sat, saturation_ceiling=ceil)
                # Preserve the legacy contract: an empty index array (not None)
                # when nothing is bad, so callers can pass it straight to pyFAI.
                cached = idx if idx is not None else np.array([], dtype=int)
                if mask_sat:
                    sat_fired = int(saturation_pixels(
                        arr0.astype(float).flatten(), ceiling=ceil).sum())
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug("frame-mask compute failed: %s", e)
                cached = None
            scan._cached_data_mask = cached
            # R3-A: the saturation mask is a default-ON behaviour change to the
            # INTEGRATION — surface it once per scan (this branch runs once;
            # later frames hit the cache).  OUTSIDE the try + guarded so the
            # advisory can never null the computed mask, and bare test holders
            # without the helper just skip it.
            if sat_fired:
                warn = getattr(self, '_warn_saturation_masked', None)
                if warn is not None:
                    warn(sat_fired, frame_size)
        return cached

    def _warn_saturation_masked(self, n_pixels: int, frame_size: int) -> None:
        """R3-A: one-time advisory that the 'Mask Saturated' policy actually
        excluded a detector-ceiling block from the INTEGRATION this run.

        The saturation mask is default-ON and changes integrated intensities,
        so a silent fire hides a real data effect.  Logged loud + surfaced in
        the GUI status line when a ``showLabel`` signal is present (absent on
        the bare test holders, so this no-ops there).  Called from the
        once-per-scan mask compute, so it fires once per run, not per frame.
        """
        pct = (100.0 * n_pixels / frame_size) if frame_size else 0.0
        msg = (f"Mask Saturated: {n_pixels} detector-ceiling pixel(s) "
               f"({pct:.2f}% of the frame) excluded from integration — a "
               f"dead/overflowed module. Untick 'Mask Saturated' to keep them.")
        logger.warning(msg)
        emit = getattr(getattr(self, 'showLabel', None), 'emit', None)
        if emit is not None:
            try:
                emit(msg)
            except Exception:
                logger.debug("showLabel emit failed for saturation advisory",
                             exc_info=True)

    def _prewarm_frame_mask(self, scan, img_data) -> None:
        """Populate ``scan._cached_data_mask`` on the main thread.

        F3 — prevents the racy initialization that happens when N
        parallel workers all simultaneously see a ``None`` cache and
        each compute + write the same mask.  Computing on the main
        thread before submitting any worker means every worker only
        ever does a cache *read* against a stable value.

        Idempotent: a no-op when the cache is already set.  Called
        from each wrangler's run loop with the first frame's
        ``img_data`` before the parallel section.
        """
        if getattr(scan, '_cached_data_mask', None) is not None:
            return
        self._resolve_frame_mask(scan, img_data)

    def _apply_threshold_inline(self, img_data):
        """Pre-clamp pixels outside the threshold band to NaN.

        Returns a fresh float32 array with out-of-band pixels
        replaced by NaN.  pyFAI's CSR integrator skips NaN pixels
        automatically (no ``dummy``/``delta_dummy`` kwargs needed),
        and NaN propagates cleanly through bg subtraction and monitor
        normalization arithmetic inside ``frame.integrate_1d/2d``.

        No-op when ``self.apply_threshold`` is False — subclasses
        that don't expose a threshold UI inherit a free pass-through.
        """
        if not self.apply_threshold:
            return img_data
        img = np.asarray(img_data, dtype=np.float32, copy=True)
        bad = (img < self.threshold_min) | (img > self.threshold_max)
        img[bad] = _THRESHOLD_NAN
        return img

    def _flush_xye_buffer(self, scan, published_idxs=None):
        """Drain ``self._xye_buffer`` and write each pending XYE file.

        Drains under :attr:`_xye_lock` so workers can keep appending
        new entries while this batch's disk IO runs.  Per-file write
        errors are logged but don't abort the batch — losing one XYE
        file shouldn't kill an otherwise-valid scan.

        P3: when ``published_idxs`` is provided, only buffer entries
        whose ``img_number`` (a.k.a. ``frame.idx``) appears in the set
        are written to disk; entries for frames that finished
        integration but never got published to the .nxs are dropped.
        This keeps the XYE directory and the .nxs frame set in sync
        after a Stop mid-batch — without the filter, in-flight
        workers that the parallel dispatcher abandoned could leave
        orphan XYE files for frames that never landed in HDF5.
        """
        with self._xye_lock:
            if not self._xye_buffer:
                return
            buf = self._xye_buffer
            self._xye_buffer = []
        if published_idxs is not None:
            published_idxs = {int(i) for i in published_idxs}
            dropped = [t for t in buf if int(t[0]) not in published_idxs]
            buf = [t for t in buf if int(t[0]) in published_idxs]
            if dropped:
                logger.info(
                    'XYE: dropped %d unpublished entries (Stop mid-batch)',
                    len(dropped),
                )
        for img_number, frame in buf:
            try:
                self.save_1d(scan, frame, img_number)
            except Exception as e:
                logger.warning(
                    'XYE write failed for frame %s: %s', img_number, e,
                )

    @staticmethod
    def save_1d(scan, frame, idx):
        """Write a single-frame XYE next to the scan's .nxs file.

        Static because it only depends on the scan + frame state —
        not on per-wrangler attributes.  Filename layout matches the
        prior imageWrangler convention so existing downstream tools
        keep working: ``<scan_dir>/<scan_name>/iq_<scan>_NNNN.xye``
        (or ``itth_...`` for 2θ units).
        """
        if frame.int_1d is None:
            return
        path = os.path.dirname(scan.data_file)
        path = os.path.join(path, scan.name)
        Path(path).mkdir(parents=True, exist_ok=True)
        r1d = frame.int_1d
        # Encode the actual 1D integration axis in the prefix so the XYE reader
        # recovers the x-axis from the name.  The old `iq if q else itth` rule
        # mislabeled every non-Q axis (GI Q_ip/Q_oop/exit) as 2θ.
        from ..display_logic import xye_prefix_for_unit
        prefix = xye_prefix_for_unit(r1d.unit)
        fname = os.path.join(
            path, f'{prefix}_{scan.name}_{str(idx).zfill(4)}.xye'
        )
        write_xye(fname, r1d.radial, r1d.intensity,
                  np.sqrt(np.abs(r1d.intensity)))

    def _save_to_disk(self, scan):
        """Persist scan state to its .nxs file (intermediate save).

        Honours the h5pool pause/resume protocol so the GUI's
        h5viewer doesn't fight the writer for the file handle, and
        the per-wrangler ``file_lock`` so reads stay quiescent
        during the write.  No-op in xye_only mode (no .nxs target).
        """
        if self.xye_only:
            return
        with self.file_lock:
            _get_h5pool().pause(scan.data_file)
            try:
                scan._save_to_nexus()
            finally:
                _get_h5pool().resume(scan.data_file)
