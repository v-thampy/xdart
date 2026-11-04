"""One fit-window controller for native hosts, on the existing worker and fit components."""
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
from pyqtgraph.Qt import QtCore

from xdart.gui.analysis.fit_export import plain_settings, write_fit_notebook
from xdart.gui.tabs.scattering.fit_view import FitPanel, FitProjection, FitSeries
from xdart.gui.tabs.scattering.results_table import peak_parameter_rows, phase_result_rows
from xdart.gui.widgets.controls_panel import FormRow
from xdart.gui.widgets.latest_request import LatestRequestOwner
from xdart.gui.pages.operation_owner import OperationTerminalStatus
from xdart.gui.pages.values import PageCleanup
from xrd_tools.analysis.fit_snapshot import FitInputSnapshot
from xrd_tools.analysis.fit_sources import phase_fit_snapshot
from xrd_tools.analysis.runner import DisplayedPeakFitAnalyzer, DisplayedPhaseFitAnalyzer, run_batch


@dataclass(frozen=True)
class FitTarget:
    """A cheap queued input; load and optional metadata resolution run on the fit worker. ``x`` is its trend
    coordinate when no trend column is chosen (an Int frame's number; None: its place in the batch). ``reference`` is
    its notebook source, or a zero-argument callable that makes it on the worker (an Int frame's row digest)."""
    label: str
    load: Callable
    trend: Callable | None = None
    column_names: tuple[str, ...] = ()
    reference: object = None
    x: float | None = None


def _reference(target):
    """A target's notebook source: its own, made on the worker when it is a callable (None: capture a sidecar)."""
    reference = target.reference if isinstance(target, FitTarget) else None
    return reference() if callable(reference) else reference


@dataclass(frozen=True)
class _Request:
    action: str
    kind: str
    scope: object
    settings: tuple
    inputs: tuple
    column: str = ''
    destination: Path | None = None
    data_root: object = None
    #: a phase Batch seeds each fit from the previous one (the analyzer's option, never a plan field)
    sequential: bool = False
    #: a Fit all run's channel (``_BatchRun``): its fitted rows and latest frame, taken at each progress poll, and its
    #: Pause and Stop
    stream: object = field(default=None, compare=False)


class _BatchRun:
    """A running Fit all's channel between the window and the worker (NEXT-128-WINDOW-POLISH item 14), a lock and
    plain values, so the worker never touches a Qt object. The worker puts each fitted frame's row ``(identity, x,
    params)`` (the maintainer's "Count + live trend") and the latest fitted frame whole with its input, which the plot
    and the table follow ("the tracked parameter updates but the 1d plot doesn't as the fits progress"); the window
    takes them at each progress poll. The window sets Pause and Stop ("add a Pause/Resume button in addition to Cancel.
    And replace Cancel with Stop"), which the worker reads between frames: a paused run waits before its next frame,
    and a stopped one fits no further frame and returns the frames it fitted, which the window keeps."""

    def __init__(self):
        self._lock = threading.Lock()
        self._rows, self._shown = [], None
        self._resume, self._stop = threading.Event(), threading.Event()
        self._resume.set()

    def put(self, row=None, shown=None):
        """A fitted frame (the worker): its row (None: it fitted no parameter) and ``(outcome, snapshot)`` to show
        (None: nothing to draw), which replaces the one not yet taken."""
        with self._lock:
            if row is not None:
                self._rows.append(row)
            if shown is not None:
                self._shown = shown

    def take(self):
        """The rows put since the last take, and the latest frame to show since then (or None)."""
        with self._lock:
            rows, shown, self._rows, self._shown = self._rows, self._shown, [], None
        return rows, shown

    @property
    def paused(self):
        return not self._resume.is_set()

    @property
    def stopped(self):
        return self._stop.is_set()

    def pause(self):
        if not self._stop.is_set():
            self._resume.clear()

    def resume(self):
        self._resume.set()

    def stop(self):
        self._stop.set()
        self._resume.set()

    def hold(self, cancel, report):
        """Before each frame (the worker): wait while paused, reporting ``"Paused"`` once it waits and ``"Fitting"``
        once it goes on; True when no further frame is fitted: stopped, or the owner's cancel (a newer request, or the
        window closing, which a paused run wakes for)."""
        if not self._resume.is_set() and not cancel.is_set() and not self._stop.is_set():
            report("Paused")
            while not self._resume.wait(.05):
                if cancel.is_set():
                    break
            if not cancel.is_set() and not self._stop.is_set():
                report("Fitting")
        return cancel.is_set() or self._stop.is_set()


#: a streaming trend redraws at most this often (s): one redraw of 3621 rows costs 11.2 ms, so a redraw per 50 ms
#: progress poll would hold the GUI (and the worker's interpreter lock) a fifth of the time
_TREND_FLUSH_S = 0.5


def _snapshot(target, settings):
    snapshot = target.load() if isinstance(target, FitTarget) else target
    if not isinstance(snapshot, FitInputSnapshot):
        raise ValueError('FIT_INPUT_REQUIRED: load a scientific input first')
    return replace(snapshot, key=replace(snapshot.key, settings=plain_settings(dict(settings))))


def _job(request, cancel, progress):
    settings = dict(request.settings)
    if request.action == 'capture':  # a template: the fit's own input, a phase fit's converted to Å⁻¹
        snapshot = _snapshot(request.inputs[0], request.settings)
        if cancel.is_set():
            return None
        return (phase_fit_snapshot(snapshot) if request.kind == 'phase' else snapshot).trace_input()
    if request.action in {'reload', 'export'}:
        snapshot = _snapshot(request.inputs[0], request.settings)
        if cancel.is_set():
            return None
        if request.action == 'export':
            def batch_inputs():
                for item in request.inputs[1:]:
                    if cancel.is_set():
                        raise ValueError('CANCELLED: notebook export cancelled')
                    reference = _reference(item)
                    yield reference if reference is not None else _snapshot(item, request.settings)
            return write_fit_notebook(request.destination, snapshot, request.kind, settings,
                                      batch_inputs=batch_inputs(), data_root=request.data_root, trend_column=request.column,
                                      reference=_reference(request.inputs[0]))
        return snapshot
    analyzer = (DisplayedPeakFitAnalyzer(settings=settings, cancel_token=cancel) if request.kind == 'peak' else
                DisplayedPhaseFitAnalyzer(settings=settings, cancel_token=cancel, sequential=request.sequential))
    last, keys, xs = [None], {}, {}
    stream, position, total = request.stream, [0], len(request.inputs)

    def load(target):
        snapshot = _snapshot(target, request.settings)
        # Only the shown final input survives; the runner retains scalar rows for earlier fits.
        last[0] = snapshot
        label = snapshot.key.result_identity
        keys[label] = snapshot.key
        xs[label] = getattr(target, 'x', None)
        if request.column and isinstance(target, FitTarget) and target.trend is not None:
            try:
                value = target.trend(request.column)
                xs[label] = value.value if value.available else None
            except (OSError, ValueError, KeyError):
                pass  # the fit remains independent of an unavailable trend coordinate
        return phase_fit_snapshot(snapshot) if request.kind == 'phase' else snapshot

    def frame_fitted(outcome):
        # its trend row as the host draws it when the batch ends: a trend column's value, else its frame, else its
        # place; and the frame to show, whole with the input it was fitted from (the runner keeps only the last whole)
        x = xs.get(outcome.label)
        row = None if not outcome.params else (
            outcome.label, x if request.column or x is not None else float(position[0]), dict(outcome.params))
        stream.put(row, None if outcome.overlay is None or last[0] is None else (outcome, last[0]))
        position[0] += 1

    # a run's Pause and Stop are read before each frame; the owner's cancel stops any request
    should_stop = cancel.is_set if stream is None else (
        lambda: stream.hold(cancel, lambda stage: progress(stage, position[0], total)))
    outcomes = run_batch(analyzer, tuple(lambda target=target: load(target) for target in request.inputs),
                         should_cancel=should_stop,
                         on_progress=lambda done, total: progress('Fitting', done, total),
                         on_frame=None if stream is None else frame_fitted, compact=True)
    return tuple(outcomes), last[0], keys, xs


_UNNAMED_CAPTURE = ("a result page's trace is not named by a NeXus file and frame; choose its file to save or export "
                    "the template")
#: a single fit's status while a batch's rows are held (NEXT-128-WINDOW-POLISH item 14, R7 F5), on both routes
BATCH_KEPT = "The trend keeps the rows from Fit all; run Fit all again to refresh them."
#: Fit all's status as it is paused or stopped and as it ends (item 14, the maintainer, 2026-11-04: "when cancelled it
#: shows "running solve is draining" in the status. why?": a solve under way is never interrupted, so a run pauses or
#: stops after the frame being fitted, and says so)
_PAUSING = "Pausing after the frame being fitted…"
_PAUSED = "Paused at {done}/{total}. Resume to continue, or Stop to keep the frames fitted."
_STOPPING = "Stopping after the frame being fitted…"
_STOPPED = "Stopped: {fitted} of {total} frames fitted"
_STOPPED_EMPTY = "Stopped before any frame was fitted"
_DONE = "Done: {fitted} of {total} frames fitted"
_RUN_WAITS = "Fit all is running; try again when it is done, or Stop it."


class SnapshotFitHost(QtCore.QObject):
    """Mount Int's fit dialogs with explicit Batch capability, never unholding Int's own profile."""
    def __init__(self, kind, parent=None, *, current, scope, batch=None, source_root=None, dialog=None,
                 owns_phase_source=True):
        super().__init__(parent)
        from .peak_fit_dialog import PeakFitDialog
        from .phase_fit_dialog import PhaseFitDialog
        self.kind, self.current, self.scope, self.batch, self.source_root = kind, current, scope, batch, source_root
        self.outcomes, self.snapshot, self._keys = (), None, {}
        self._last_settings = ()
        self._batching = False
        #: whether shutdown also stops the Fitter's Add CIF… worker (Int's page owns it for the windows it mounts)
        self._owns_phase_source = bool(owns_phase_source)
        self.owner = LatestRequestOwner(_job, self._done, same=lambda a, b: a is b, progress=self._progress,
                                        parent=self)
        self._shown_outcome = None
        #: the adopted fits' trend rows (a cancelled or refused Batch's streamed rows give way to them again); a running
        #: Batch's rows taken but not yet drawn, its request, and when the trend last drew them
        self._trend_rows, self._trend_phases = [], ()
        #: whether the adopted rows are a batch's: a single fit then is shown and never changes them (R7 F5)
        self._batch_held = False
        self._batch_scope = None
        self._held_rows, self._held_shown, self._held_request, self._last_flush = [], None, None, 0.0
        #: a running Fit all's latest count (Resume shows it again)
        self._run_count = None
        self._flush_timer = QtCore.QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.timeout.connect(self._flush_trend)
        if dialog is not None:
            # NEXT-128-WINDOW-POLISH item 11: Int's page mounts its own window and routes its actions; this host runs
            # the window's file-source fits, its batches and its exports, with none of the result pages' rows. Its
            # Fit all (and Stop) reaches ``_batch`` through the page; its Pause needs no page gate.
            self.dialog, self.column_row = dialog, None
            dialog.fit_view.components_toggle.toggled.connect(lambda _on: self._draw_outcome(self._shown_outcome))
            dialog.pause_btn.clicked.connect(self._pause)
            return
        self.dialog = (PeakFitDialog if kind == 'peak' else PhaseFitDialog)(parent=parent,
            vnext_submit=self._action, allow_batch=batch is not None)
        self.dialog.batch_btn.clicked.connect(self._batch)
        self.dialog.pause_btn.clicked.connect(self._pause)
        self.dialog.trend.save.clicked.disconnect()
        self.dialog.trend.save.clicked.connect(self._choose_csv)
        self.dialog.export_actions.export = self._export
        self.dialog.export_actions.source_root = source_root() if callable(source_root) else source_root
        self.dialog._export_snapshot_provider = self.current
        self.column_row = FormRow(label='Trend column', path=('fit', 'trend_column'), value='Result order',
                                   kind='combo', choices=('Result order',))
        self.column_row.editor.setEditable(True)
        self.column_row.setToolTip('Contributing-row mean of this scan column; missing values are unavailable')
        self.dialog.busy_probe = lambda: self.computing  # Load settings waits for a fit (both tools)
        if kind == 'phase':  # Phase Analysis keeps it in its Batch card
            self.dialog.add_batch_row(self.column_row)
        else:
            self.dialog.layout().insertWidget(0, self.column_row)
        self.column_row.setVisible(batch is not None)
        # a peak fit's own peaks are evaluated only while Components is on
        self.dialog.fit_view.components_toggle.toggled.connect(lambda _on: self._draw_outcome(self._shown_outcome))

    @property
    def computing(self):
        return self.owner.computing

    @property
    def trend_column(self):
        if self.column_row is None:
            return ''
        value = self.column_row.editor.currentText().strip()
        return '' if value == 'Result order' else value

    def set_trend_columns(self, names):
        if self.column_row is None:
            return
        combo = self.column_row.editor
        current = combo.currentText()
        combo.clear()
        combo.addItems(('Result order', *dict.fromkeys(str(n) for n in names)))
        combo.setCurrentText(current)

    def _request(self, action, inputs, destination=None):
        source = getattr(self.dialog, 'source_row', None)
        reason = '' if source is None else source.input_refusal()
        if reason:
            self.dialog.status.setText(reason)
            source.report_problem(reason)
            return
        if self._batching and action != 'batch':  # Fit all owns the worker until it ends (an export would replace it)
            self.dialog.status.setText(_RUN_WAITS)
            return
        settings = self._request_settings(action)
        self.owner.forget()
        self.dialog.status.setText({'reload': 'Loading…', 'export': 'Exporting…',
                                    'capture': 'Capturing the substrate…'}.get(action, 'Fitting…'))
        self._batching = action == 'batch'
        self._end_stream()
        if action == 'batch':
            self.dialog.set_batch_running(True)
            self.dialog.trend.clear()  # the batch's own rows join as its frames are fitted
        elif action == 'fit' and source is not None:
            source.set_busy('A fit is running; step frames when it is done.')
        elif action == 'reload':
            self.dialog.fit_btn.setEnabled(False)
        sequential = action == 'batch' and self.kind == 'phase' and self.dialog.sequential_batch()
        self.owner.ask(_Request(action, self.kind, self.scope(), settings, tuple(inputs),
                                self.trend_column, destination,
                                self.source_root() if callable(self.source_root) else self.source_root,
                                sequential=sequential, stream=_BatchRun() if action == 'batch' else None))

    def _progress(self, request, progress):
        """A running Fit all's count in the status (or its Pause or Stop), its fitted frames into the trend and its
        latest one into the plot and the table (NEXT-128-WINDOW-POLISH item 14: "i started a batch fit over a long
        sequence. there is no update progress"; his "Count + live trend"; "the tracked parameter updates but the 1d plot
        doesn't as the fits progress")."""
        if self.dialog is None or request.action != 'batch' or request.stream is None:
            return
        run = request.stream
        self._run_count = (progress.completed, progress.total)
        if run.stopped:
            self.dialog.status.setText(_STOPPING)
        elif run.paused:
            self.dialog.status.setText(_PAUSED.format(done=progress.completed, total=progress.total)
                                       if progress.stage == 'Paused' else _PAUSING)
        else:
            self.dialog.set_batch_progress(progress.completed, progress.total)
        rows, shown = run.take()
        self._held_rows += rows
        if shown is not None:
            self._held_shown = shown
        self._held_request = request
        self._flush_trend()

    def _flush_trend(self):
        """Draw the held rows and the latest fitted frame: at once, or when ``_TREND_FLUSH_S`` has passed since the
        last draw (a timer)."""
        request = self._held_request
        if request is None or self.dialog is None or self.owner.wanted is not request:
            self._end_stream()
            return
        if not self._held_rows and self._held_shown is None:
            return
        wait = self._last_flush + _TREND_FLUSH_S - time.monotonic()
        if wait > 0:
            if not self._flush_timer.isActive():
                self._flush_timer.start(int(wait * 1000) + 1)
            return
        rows, shown = self._held_rows, self._held_shown
        self._held_rows, self._held_shown = [], None
        self._last_flush = time.monotonic()
        if rows:
            self.dialog.trend.set_rows(rows, phase_names=dict(request.settings).get('phase_names', ()))
        if shown is not None:
            self._show_fitted(*shown)

    def _show_fitted(self, outcome, snapshot):
        """A running Fit all's latest fitted frame: its points, title and fit in the plot and its parameters in the
        table, drawn without the trace read's option writes (``set_vnext_trace`` writes typed range bounds and the
        uncertainty switch, and a run adopts its results only with the options it started with); the run's end
        shows its last frame through the full read."""
        from .peak_fit_dialog import show_fitted_trace
        self.snapshot = snapshot
        show_fitted_trace(self.dialog, snapshot.trace_input())
        self._shown_outcome = outcome
        self._draw_outcome(outcome)
        if self.kind == 'phase':
            self.dialog.show_phase_rows(outcome.result)
        else:
            self.dialog.table.show_rows(*peak_parameter_rows(outcome.result))

    def _end_stream(self):
        """No Batch streams now: its held rows and frame go undrawn."""
        self._flush_timer.stop()
        self._held_rows, self._held_shown, self._held_request, self._last_flush = [], None, None, 0.0
        self._run_count = None

    def _restore_trend(self):
        """A Batch ended unadopted (cancelled, refused or failed): the trend shows the adopted fits' rows again."""
        if self.dialog is not None:
            self.dialog.trend.clear()
            self.dialog.trend.set_rows(self._trend_rows, phase_names=self._trend_phases)

    def _request_settings(self, action):
        # Frozen phase entries feed the analyzer unchanged; only FitInputKey
        # uses their detached plain identity. Notebook requests use CIF paths.
        values = (self.dialog.export_actions.settings() if action == 'export'
                  else self.dialog.vnext_fit_values())
        return tuple(sorted(values.items()))

    def reload(self):
        try:
            target = self.current()
            if target is None:
                raise ValueError('FIT_INPUT_REQUIRED: select a result')
            self.invalidate()
            self._request('reload', (target,))
        except Exception as error:
            self.dialog.status.setText(str(error))

    def _action(self, action, _values):
        if action == 'close':
            self.invalidate()
        elif action == 'reload':
            self.reload()
        elif action in {'fit', 'capture_template'}:
            try:
                self._request('fit' if action == 'fit' else 'capture', (self.current(),))
            except Exception as error:
                self.dialog.status.setText(str(error))

    def _running_run(self):
        """The running Fit all's channel, or None."""
        request = self.owner.wanted
        return request.stream if self._batching and request is not None and request.action == 'batch' else None

    def _batch(self):
        run = self._running_run()
        if run is not None:
            # Stop (item 14: "replace Cancel with Stop"; "when the batch fit is cancelled it clears the parameter
            # tracking window - it shouldn't"): the run ends after the frame being fitted and ``_done`` keeps every
            # frame it fitted
            run.stop()
            self.dialog.set_batch_stopping()
            self.dialog.status.setText(_STOPPING)
            return
        if self.computing and self.column_row is not None:  # a result page's single fit: dropped, as before
            self.owner.release()
            self._end_stream()
            self.dialog.set_batch_running(False)
            self.dialog.status.setText('Fit cancelled')
            return
        refusal = self.dialog.fit_refusal()  # both windows name why their options cannot fit (item 14)
        if refusal:
            self.dialog.status.setText(refusal)
            return
        if self.batch is not None:
            try:
                targets = tuple(self.batch())
            except (ValueError, OSError) as error:  # the host names why it has nothing to batch
                self.dialog.status.setText(str(error))
                return
            self.set_trend_columns(n for target in targets for n in target.column_names)
            self._request('batch', targets)

    def _pause(self):
        """Pause or Resume a running Fit all (item 14: "add a Pause/Resume button in addition to Cancel"): paused, it
        waits after the frame being fitted until Resume, Stop or the window closing."""
        run = self._running_run()
        if self.dialog is None or run is None or run.stopped:
            return
        if run.paused:
            run.resume()
            self.dialog.set_batch_paused(False)
            if self._run_count is not None:
                self.dialog.set_batch_progress(*self._run_count)
        else:
            run.pause()
            self.dialog.set_batch_paused(True)
            self.dialog.status.setText(_PAUSING)

    def _done(self, request, terminal):
        self._batching = False
        self._end_stream()
        if self.dialog is None:  # a retired host: its window is gone
            return False
        self.dialog.set_batch_running(False)
        try:
            # plain projections on both sides: a substrate compares by its trace identity, never its arrays
            unchanged = (request.scope == self.scope()
                         and plain_settings(dict(request.settings))
                         == plain_settings(dict(self._request_settings(request.action))))
        except (ValueError, KeyError, OSError):
            unchanged = False
        if not unchanged:
            if request.action == 'batch':
                self._restore_trend()
            self.dialog.status.setText('FIT_INPUT_CHANGED: reload the current selection')
            return False
        if terminal.status is not OperationTerminalStatus.RETURNED:
            if request.action == 'batch':
                self._restore_trend()
            message = terminal.failure_message or terminal.status.value
            self.dialog.status.setText(message)
            source = getattr(self.dialog, 'source_row', None)
            if source is not None and 'FIT_SOURCE_' in message:
                source.report_problem(message)
            return False
        if request.action == 'export':
            self.dialog.status.setText(f'Exported {terminal.payload}')
            return True
        if request.action == 'capture':  # a result's trace names no NeXus frame: usable, not saved or exported
            return terminal.payload is not None and self.dialog.capture_background_template(
                terminal.payload, None, _UNNAMED_CAPTURE)
        if request.action == 'reload':
            self.snapshot = terminal.payload
            if self.snapshot is None:
                return False
            self._show_trace()
            return True
        outcomes, snapshot, keys, xs = terminal.payload
        run = request.stream if request.action == 'batch' else None
        stopped = run is not None and run.stopped and len(outcomes) < len(request.inputs)
        if stopped and not any(each.ok for each in outcomes):
            # Stop before any frame was fitted (item 14): nothing replaces the rows held before the run
            self._restore_trend()
            self.dialog.status.setText(_STOPPED_EMPTY)
            return False
        self.snapshot = snapshot
        # a single fit while a batch's rows are held (NEXT-128-WINDOW-POLISH item 14, Codex R7 F5) is shown and changes
        # none of them: its input (a stored curve, or the trace Int shows) and settings need not be the batch's, so
        # one table would mix them; the next Batch replaces the rows (Reload or another source clears them)
        kept = self._batch_held and request.action != 'batch'
        self._adopt_template(outcomes)
        if not kept:
            self.outcomes, self._keys = tuple(outcomes), keys
            self._trend_rows = [(each.label, x if request.column or x is not None else float(position), each.params)
                                for position, each in enumerate(outcomes) for x in (xs.get(each.label),)]
            self._trend_phases = tuple(dict(request.settings).get('phase_names', ()))
            self._batch_held = request.action == 'batch'
            self._batch_scope = request.scope if self._batch_held else None
            # Binding the template's content identity is a result of this fit, not a settings edit: the CSV names the
            # bound form (Codex R7 F2)
            self._last_settings = plain_settings(dict(self._request_settings(request.action)))
            self.dialog.trend.clear()
            # every row in one redraw (item 14): one ``set_row`` per outcome redrew every row each time
            self.dialog.trend.set_rows(self._trend_rows, phase_names=self._trend_phases)
        if not outcomes:
            return False
        outcome = outcomes[-1]
        self._shown_outcome = None
        if outcome.overlay is not None and self.snapshot is not None:
            self._show_trace()
            self._shown_outcome = outcome
            self._draw_outcome(outcome)
            if self.kind == 'phase':
                self.dialog.show_phase_result_details(outcome.result)
            else:
                self.dialog.table.show_rows(*peak_parameter_rows(outcome.result))
        if request.action == 'batch':  # where the run stands at its end (item 14), a stopped one's frames kept
            fitted = sum(1 for each in self.outcomes if each.ok)
            self.dialog.status.setText((_STOPPED if stopped else _DONE).format(fitted=fitted,
                                                                                total=len(request.inputs)))
        elif self.kind != 'phase' or outcome.overlay is None:
            self.dialog.status.setText(outcome.message or ('Fit complete' if outcome.ok else 'Fit refused'))
        if kept:
            self.note_batch_kept()
        source = getattr(self.dialog, 'source_row', None)
        if source is not None and 'FIT_SOURCE_' in (outcome.message or ''):
            source.report_problem(outcome.message)
        return outcome.ok

    @property
    def holds_batch(self):
        """Whether the adopted rows are a batch's, which a single fit leaves as they are (R7 F5)."""
        return self._batch_held

    def batch_scope_changed(self):
        """Whether held rows belong to another host scope (Int can Browse a new source without a Source-card edit)."""
        return self._batch_held and self._batch_scope != self.scope()

    def note_batch_kept(self):
        """Say in the status that the trend keeps the batch's rows (a single fit's, on either route)."""
        if self.dialog is not None:
            text = self.dialog.status.text().rstrip(". ")
            self.dialog.status.setText(f"{text}. {BATCH_KEPT}" if text else BATCH_KEPT)

    def _draw_outcome(self, outcome):
        """The shown fit over the pattern (a phase fit's x back in a 2θ result's own axis); a peak fit's own peaks
        only while Components is on."""
        if outcome is None or outcome.overlay is None or self.snapshot is None:
            return
        overlay = outcome.overlay
        if self.kind == 'phase' and self.snapshot.axes[0].unit in {'2th_deg', '2th_rad'}:
            from xrd_tools.core.containers import convert_radial_axis
            convert = lambda values: convert_radial_axis(np.asarray(values), 'q_A^-1',
                self.snapshot.axes[0].unit, self.snapshot.wavelength_angstrom)
            x, markers = convert(overlay.x), convert(overlay.markers)
        else:
            x, markers = overlay.x, overlay.markers
        upper = tuple(FitSeries(x, values, name, name) for name, values in overlay.traces.items()
                      if name != 'residual')
        result = outcome.result
        if (self.kind != 'phase' and getattr(result, 'model', '')
                and self.dialog.fit_view.components_toggle.isChecked()):
            from xrd_tools.analysis.fitting.fit import peak_component_curves
            upper += tuple(FitSeries(x, values, 'component', name) for name, values in peak_component_curves(
                result.model, result.parameter_names, result.parameter_values, overlay.x))
        lower = (FitSeries(x, overlay.traces['residual'], 'residual', 'residual'),)
        self.dialog.fit_view.reconcile(FitProjection(self.dialog._pattern_panel(*upper, markers=tuple(markers)),
                                                    FitPanel(lower, zero_line=True)))

    def _adopt_template(self, outcomes):
        """A fit's template echo on the shared row (NEXT-128-PEAK-FIT): the first completed fit of the row's file
        records its identity; else a template refusal says why."""
        row = self.dialog.background_template_row
        results = [outcome.result for outcome in outcomes if getattr(outcome, 'result', None) is not None]
        if not any(row.adopt(result) for result in results) and results:
            row.refuse(results[-1])

    def _show_trace(self):
        options = {'wavelength': self.snapshot.wavelength_angstrom} if self.kind == 'phase' else {}
        self.dialog.set_vnext_trace(self.snapshot.trace_input(), **options)

    def invalidate(self):
        running = self._batching
        self.owner.release()
        self._batching = False
        self._end_stream()
        self.snapshot = self._shown_outcome = None
        self.outcomes, self._keys, self._trend_rows, self._trend_phases = (), {}, [], ()
        self._batch_held = False
        self._batch_scope = None
        if self.dialog is not None:
            if running:  # a dropped run's Stop, Pause and locks go with it
                self.dialog.set_batch_running(False)
            self.dialog.clear_input()
            if self.dialog.source_row is not None:
                self.dialog.source_row.set_busy('')
            self.dialog.trend.clear()

    def forget_shown(self):
        """A step to another frame of the same result (NEXT-128-WINDOW-POLISH item 14): the shown frame and its fit go
        and a request still computing is dropped; the batch's rows and trend stay, since they cover every frame."""
        running = self._batching
        self.owner.release()
        self._batching = False
        self._end_stream()
        self.snapshot = self._shown_outcome = None
        if self.dialog is not None:
            if running:  # a dropped run's Stop, Pause and locks go with it
                self.dialog.set_batch_running(False)
            self.dialog.clear_input()
            if self.dialog.source_row is not None:
                self.dialog.source_row.set_busy('')

    def read_frame(self):
        """Read the current input (the stepped frame) with what is shown forgotten first (``forget_shown``), the
        batch's rows and trend kept: Reload's read without Reload's invalidation."""
        self.forget_shown()
        try:
            target = self.current()
            if target is None:
                raise ValueError('FIT_INPUT_REQUIRED: select a result')
            self._request('reload', (target,))
        except Exception as error:
            self.dialog.status.setText(str(error))

    def retire(self):
        """The window closed (Int's page): drop its fit and the window, keeping the worker for ``shutdown``."""
        self.invalidate()
        self.dialog = None

    def _export(self, path):
        try:
            current = self.current()
            targets = () if self.batch is None else tuple(self.batch())
            self._request('export', (current, *targets), Path(path))
        except Exception as error:
            self.dialog.status.setText(str(error))

    def export_to(self, path):
        """Direct publication interface, also used by scripts/tests; GUI uses the worker path."""
        settings = self.dialog.export_actions.settings()
        frozen = plain_settings(settings)
        inputs = () if self.batch is None else tuple(self.batch())
        batch = (_reference(item) or _snapshot(item, frozen) for item in inputs)
        current = self.current()
        return write_fit_notebook(path, _snapshot(current, frozen), self.kind, settings,
                                  batch_inputs=batch, trend_column=self.trend_column,
                                  data_root=self.source_root() if callable(self.source_root) else self.source_root,
                                  reference=_reference(current))

    def export_csv(self, path):
        from xrd_tools.analysis.fit_table import outcomes_table, table_csv
        unit = '' if self.snapshot is None else self.snapshot.axes[0].unit
        table = outcomes_table(self.outcomes, axis_unit=unit, contexts=self._keys,
                              model=self.kind, settings=dict(self._last_settings))
        Path(path).write_text(table_csv(table), encoding='utf-8')

    def _choose_csv(self):
        from xdart.gui.widgets.file_dialogs import browse
        path = '' if self.snapshot is None else self.snapshot.key.path
        picks = browse(self.dialog, 'Save fit results', save=True, current=path,
                       default_name='fit.csv', file_filter='CSV (*.csv)')
        if picks:
            try:
                self.export_csv(picks[0])
                self.dialog.status.setText(f'Saved {picks[0]}')
            except Exception as error:
                self.dialog.status.setText(str(error))

    def shutdown(self):
        """Close the fit worker and the Fitter's Add CIF… worker: CLEAN, or PENDING while either runs."""
        self.invalidate()
        receipts = [self.owner.shutdown()]
        source = getattr(self.dialog, 'phase_source', None) if self._owns_phase_source else None
        if source is not None:
            receipts.append(source.shutdown())
        return next((receipt for receipt in receipts if receipt.status is not PageCleanup.CLEAN), receipts[0])
