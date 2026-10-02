"""One fit-window controller for native hosts, on the existing worker and fit components."""
from dataclasses import dataclass, replace
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
from xrd_tools.analysis.fit_snapshot import FitInputSnapshot
from xrd_tools.analysis.fit_sources import phase_fit_snapshot
from xrd_tools.analysis.runner import DisplayedPeakFitAnalyzer, DisplayedPhaseFitAnalyzer, run_batch


@dataclass(frozen=True)
class FitTarget:
    """A cheap queued input; load and optional metadata resolution run on the fit worker."""
    label: str
    load: Callable
    trend: Callable | None = None
    column_names: tuple[str, ...] = ()
    reference: object = None


@dataclass(frozen=True)
class _Request:
    action: str
    kind: str
    scope: object
    settings: tuple
    inputs: tuple
    column: str = ''
    destination: Path | None = None


def _snapshot(target, settings):
    snapshot = target.load() if isinstance(target, FitTarget) else target
    if not isinstance(snapshot, FitInputSnapshot):
        raise ValueError('FIT_INPUT_REQUIRED: load a scientific input first')
    return replace(snapshot, key=replace(snapshot.key, settings=settings))


def _job(request, cancel, progress):
    settings = dict(request.settings)
    if request.action in {'reload', 'export'}:
        snapshot = _snapshot(request.inputs[0], request.settings)
        if cancel.is_set():
            return None
        if request.action == 'export':
            refs = tuple(item.reference for item in request.inputs[1:]
                         if isinstance(item, FitTarget) and item.reference is not None)
            return write_fit_notebook(request.destination, snapshot, request.kind, settings, batch=refs)
        return snapshot
    analyzer_type = DisplayedPeakFitAnalyzer if request.kind == 'peak' else DisplayedPhaseFitAnalyzer
    analyzer = analyzer_type(settings=settings, cancel_token=cancel)
    last, keys, xs = [None], {}, {}

    def load(target):
        snapshot = _snapshot(target, request.settings)
        # Only the shown final input survives; the runner retains scalar rows for earlier fits.
        last[0] = snapshot
        label = snapshot.key.result_identity
        keys[label] = snapshot.key
        xs[label] = None
        if request.column and isinstance(target, FitTarget) and target.trend is not None:
            try:
                value = target.trend(request.column)
                xs[label] = value.value if value.available else None
            except (OSError, ValueError, KeyError):
                pass  # the fit remains independent of an unavailable trend coordinate
        return phase_fit_snapshot(snapshot) if request.kind == 'phase' else snapshot

    outcomes = run_batch(analyzer, tuple(lambda target=target: load(target) for target in request.inputs),
                         should_cancel=cancel.is_set,
                         on_progress=lambda done, total: progress('Fitting', done, total), compact=True)
    return tuple(outcomes), last[0], keys, xs


class SnapshotFitHost(QtCore.QObject):
    """Mount Int's fit dialogs with explicit Batch capability, never unholding Int's own profile."""
    def __init__(self, kind, parent=None, *, current, scope, batch=None, source_root=None):
        super().__init__(parent)
        from .peak_fit_dialog import PeakFitDialog
        from .phase_fit_dialog import PhaseFitDialog
        self.kind, self.current, self.scope, self.batch, self.source_root = kind, current, scope, batch, source_root
        self.outcomes, self.snapshot, self._keys = (), None, {}
        self._last_settings = ()
        self.dialog = (PeakFitDialog if kind == 'peak' else PhaseFitDialog)(parent=parent,
            vnext_submit=self._action, allow_batch=batch is not None)
        self.dialog.batch_btn.clicked.connect(self._batch)
        self.dialog.trend.save.clicked.disconnect()
        self.dialog.trend.save.clicked.connect(self._choose_csv)
        self.dialog.export_actions.export = self._export
        self.dialog._export_snapshot_provider = self.current
        self.column_row = FormRow(label='Trend column', path=('fit', 'trend_column'), value='Result order',
                                   kind='combo', choices=('Result order',))
        self.column_row.editor.setEditable(True)
        self.column_row.setToolTip('Contributing-row mean of this scan column; missing values are unavailable')
        self.dialog.layout().insertWidget(0, self.column_row)
        self.column_row.setVisible(batch is not None)
        self.owner = LatestRequestOwner(_job, self._done, same=lambda a, b: a is b, parent=self)

    @property
    def computing(self):
        return self.owner.computing

    @property
    def trend_column(self):
        value = self.column_row.editor.currentText().strip()
        return '' if value == 'Result order' else value

    def set_trend_columns(self, names):
        combo = self.column_row.editor
        current = combo.currentText()
        combo.clear()
        combo.addItems(('Result order', *dict.fromkeys(str(n) for n in names)))
        combo.setCurrentText(current)

    def _request(self, action, inputs, destination=None):
        settings = plain_settings(self.dialog.vnext_fit_values())
        self.owner.forget()
        self.dialog.status.setText('Loading…' if action == 'reload' else 'Exporting…' if action == 'export' else 'Fitting…')
        if action == 'batch':
            self.dialog.set_batch_running(True)
        elif action == 'reload':
            self.dialog.fit_btn.setEnabled(False)
        self.owner.ask(_Request(action, self.kind, self.scope(), settings, tuple(inputs), self.trend_column, destination))

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
        elif action == 'fit':
            try:
                self._request('fit', (self.current(),))
            except Exception as error:
                self.dialog.status.setText(str(error))

    def _batch(self):
        if self.computing:
            self.owner.release()
            self.dialog.set_batch_running(False)
            self.dialog.status.setText('Cancelled; the running solve is draining')
            return
        if self.batch is not None:
            targets = tuple(self.batch())
            self.set_trend_columns(n for target in targets for n in target.column_names)
            self._request('batch', targets)

    def _done(self, request, terminal):
        self.dialog.set_batch_running(False)
        try:
            unchanged = (request.scope == self.scope()
                         and request.settings == plain_settings(self.dialog.vnext_fit_values()))
        except (ValueError, KeyError, OSError):
            unchanged = False
        if not unchanged:
            self.dialog.status.setText('FIT_INPUT_CHANGED: reload the current selection')
            return False
        if terminal.status is not OperationTerminalStatus.RETURNED:
            self.dialog.status.setText(terminal.failure_message or terminal.status.value)
            return False
        if request.action == 'export':
            self.dialog.status.setText(f'Exported {terminal.payload}')
            return True
        if request.action == 'reload':
            self.snapshot = terminal.payload
            if self.snapshot is None:
                return False
            self.dialog.set_vnext_trace(self.snapshot.trace_input())
            return True
        self.outcomes, self.snapshot, self._keys, xs = terminal.payload
        self._last_settings = request.settings
        self.dialog.trend.clear()
        for position, outcome in enumerate(self.outcomes):
            self.dialog.trend.set_row(outcome.label, xs.get(outcome.label) if request.column else float(position), outcome.params)
        if not self.outcomes:
            return False
        outcome = self.outcomes[-1]
        if outcome.overlay is not None and self.snapshot is not None:
            self.dialog.set_vnext_trace(self.snapshot.trace_input())
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
            lower = (FitSeries(x, overlay.traces['residual'], 'residual', 'residual'),)
            self.dialog.fit_view.reconcile(FitProjection(self.dialog._pattern_panel(*upper, markers=tuple(markers)),
                                                        FitPanel(lower, zero_line=True)))
            self.dialog.table.show_rows(*(peak_parameter_rows(outcome.result) if self.kind == 'peak'
                                          else phase_result_rows(outcome.result)))
        self.dialog.status.setText(outcome.message or ('Fit complete' if outcome.ok else 'Fit refused'))
        return outcome.ok

    def invalidate(self):
        self.owner.release()
        self.snapshot = None
        self.outcomes, self._keys = (), {}
        self.dialog.clear_input()
        self.dialog.trend.clear()

    def _export(self, path):
        try:
            current = self.current()
            targets = () if self.batch is None else tuple(self.batch())
            self._request('export', (current, *targets), Path(path))
        except Exception as error:
            self.dialog.status.setText(str(error))

    def export_to(self, path):
        """Direct publication interface, also used by scripts/tests; GUI uses the worker path."""
        settings = self.dialog.vnext_fit_values()
        return write_fit_notebook(path, _snapshot(self.current(), plain_settings(settings)), self.kind, settings)

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
        self.invalidate()
        return self.owner.shutdown()
