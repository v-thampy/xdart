"""Native RSM plane/box fitting on Int's shared controls, scientific view and worker.

Only admitted scientific payloads enter this window. Extraction, fitting and
model evaluation run on LatestRequestOwner; display Log never enters a request.
"""
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import numpy as np
from pyqtgraph.Qt import QtWidgets

from xdart.gui.pages.operation_owner import OperationCancelled, OperationTerminalStatus
from xdart.gui.analysis.fit_export import FitExportActions, write_fit_notebook
from xdart.gui.analysis.snapshot_fit_host import FitTarget
from xdart.gui.tabs.scattering.advanced_editor import AdvancedDialog
from xdart.gui.tabs.scattering.results_table import ResultsTable
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_values import (
    AxisProjection, KeyedImageProjection, KeyedPanelsProjection, RectRegion, ShellCommandKind,
)
from xdart.gui.tabs.scattering.trend_view import TrendView
from xdart.gui.widgets.controls_panel import ActionButton, FormRow, RangeRow, SectionCard
from xdart.gui.widgets.latest_request import LatestRequestOwner
from xrd_tools.analysis.fit_snapshot import FitInputKey
from xrd_tools.analysis.fit_table import outcomes_table, table_csv
from xrd_tools.analysis.fit_sources import rsm_reciprocal_transform, rsm_region_snapshot
from xrd_tools.analysis.fitting.region import MAX_POINTS, PeakSeed, RegionAxis
from xrd_tools.analysis.fitting.region_analyzers import (
    REGION_2D_KEY, REGION_3D_KEY, Region2DFitAnalyzer, Region2DInput, Region3DFitAnalyzer, Region3DInput,
)
from xrd_tools.analysis.runner import AnalysisInput
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, AnalysisArtifactPayload
from xrd_tools.notebooks.inputs import SourceReference
from xrd_tools.session.display_logic import PanelKey, PanelRole
from xrd_tools.session.readiness import ControlAction, ControlActionSpec, SectionId


_KEYS = tuple(PanelKey(PanelRole.SLICE_2D, name) for name in ('data', 'model', 'residual'))


@dataclass(frozen=True, eq=False)
class _RegionRequest:
    payload: AnalysisArtifactPayload
    key: FitInputKey
    ndim: int
    bounds: tuple
    plane: str
    index: int
    seeds: tuple
    settings: tuple
    solve: bool = False
    snapshot: object = None
    outcome: object = None
    destination: object = None
    source_root: object = None

    def matches(self, other):
        return (other is not None and self.key == other.key and self.plane == other.plane
                and self.index == other.index and self.solve == other.solve and self.outcome is other.outcome
                and self.destination == other.destination)


@dataclass(frozen=True)
class _AxisUnit:
    name: str
    unit: str


@dataclass(frozen=True)
class _TableContext:
    key: FitInputKey
    axes: tuple


def _reference(request):
    selected = (('bounds', request.bounds),)
    if request.ndim == 2:
        selected += (('plane', request.plane), ('index', request.index))
    payload = request.payload
    return SourceReference(request.key.path, 'rsm_region', request.key.result_identity,
        payload.result_fingerprint, payload.inspection.request_fingerprint, selection=selected,
        original_path=request.key.path)


def _settings(request):
    return dict(request.settings, seeds=tuple((seed.id, seed.center, seed.fwhm, seed.height) for seed in request.seeds))


def _region_job(request, cancel, _progress):
    if cancel.is_set():
        raise OperationCancelled()
    snapshot = request.snapshot or rsm_region_snapshot(request.payload, request.key, bounds=request.bounds,
        plane=request.plane if request.ndim == 2 else None, index=request.index if request.ndim == 2 else None)
    if request.destination is not None:
        if cancel.is_set():
            raise OperationCancelled()
        return write_fit_notebook(request.destination, snapshot, f'region{request.ndim}d', _settings(request),
            reciprocal_transform=rsm_reciprocal_transform(request.payload) if request.ndim == 3 else None,
            reference=_reference(request), data_root=request.source_root)
    outcome, analysis = request.outcome, None
    if request.solve:
        axes = tuple(RegionAxis(axis.name, axis.values, axis.unit) for axis in snapshot.axes)
        mask = None if snapshot.mask is None else ~snapshot.mask
        settings = dict(request.settings)
        if request.ndim == 2:
            region = Region2DInput(snapshot.values, *axes, mask=mask, sigma=snapshot.sigma)
            analyzer = Region2DFitAnalyzer(seeds=request.seeds, cancel_token=cancel, **settings)
            metadata = {REGION_2D_KEY: region}
        else:
            region = Region3DInput(snapshot.values, axes, mask=mask, sigma=snapshot.sigma,
                                  reciprocal_transform=rsm_reciprocal_transform(request.payload))
            analyzer = Region3DFitAnalyzer(seeds=request.seeds, cancel_token=cancel,
                                           **{k: v for k, v in settings.items() if k != 'profile'})
            metadata = {REGION_3D_KEY: region}
        analysis = analyzer.analyze(AnalysisInput(request.key.result_identity, np.empty(0), np.empty(0),
                                                  metadata=metadata))
        if not analysis.ok:
            raise ValueError(analysis.message)
        outcome = analysis.result.payload
    if cancel.is_set():
        raise OperationCancelled()
    if request.ndim == 3:
        normal = next(i for i, axis in enumerate(snapshot.axes) if axis.name == request.plane)
        index = snapshot.values.shape[normal]//2 if request.index < 0 else request.index
        if not 0 <= index < snapshot.values.shape[normal]:
            raise ValueError('FIT_SLICE_INDEX_INVALID')
        axes = tuple(axis for i, axis in enumerate(snapshot.axes) if i != normal)
        data = np.take(snapshot.values, index, axis=normal)
        comparison = outcome.comparison_slice(snapshot.values, request.plane, index) if outcome else None
        captions = tuple(f'{value:.6g}' for value in snapshot.axes[normal].values)
    else:
        index, captions, axes, data = 0, (), snapshot.axes, snapshot.values
        comparison = outcome.comparison(snapshot.values) if outcome else None
    images = (comparison.data, comparison.model, comparison.residual) if comparison else (
        data, np.full(data.shape, np.nan), np.full(data.shape, np.nan))
    panels = tuple(KeyedImageProjection(key, key.instance.title(), image.T,
        AxisProjection(axes[0].values, axes[0].name, axes[0].unit),
        AxisProjection(axes[1].values, axes[1].name, axes[1].unit), index, captions,
        navigation=request.ndim == 3, presentation='signed' if key.instance == 'residual' else 'shared')
        for key, image in zip(_KEYS, images))
    if cancel.is_set():
        raise OperationCancelled()
    return snapshot, outcome, analysis, panels


class RegionFitWindow(QtWidgets.QDialog):
    """A shared 2-D/3-D fit window; the host supplies its admitted artifact, never viewer pixels."""

    def __init__(self, parent=None, ndim=2):
        if ndim not in (2, 3):
            raise ValueError('a region fit is 2-D or 3-D')
        super().__init__(parent)
        self.ndim = ndim
        self.setModal(False)
        self.setObjectName(f'region{ndim}DFitWindow')
        self.setWindowTitle(f'{ndim}-D Region Fit')
        outer = QtWidgets.QVBoxLayout(self)
        layout = QtWidgets.QHBoxLayout()
        outer.addLayout(layout)
        self.view = ScientificView()
        self.view.set_hidden_tools('norm', 'background', 'raw', 'raw_pane', 'slice')
        self.view.commandRequested.connect(self._view_command)
        self.view.keyedRegionChanged.connect(self._region_changed)
        layout.addWidget(self.view, 1)
        side = QtWidgets.QVBoxLayout()
        layout.addLayout(side)
        self.card = SectionCard(f'{ndim}-D Region Fit — recorded volume', collapsible=False)
        self.card.setToolTip('Uses the recorded linear volume; the RSM main view\'s Set BG is not applied.')
        side.addWidget(self.card)
        self.status = self.card.status
        self.rows, self.range_rows = {}, {}
        self._add('plane', 'Plane normal', '', kind='combo')
        if ndim == 2:
            self._add('index', 'Slice index', '0', kind='int', minimum=0)
        self._add('components', 'Components', '1', kind='int', minimum=1, maximum=3)
        self._add('background', 'Background', 'linear', kind='combo', choices=('linear', 'constant', 'none'))
        if ndim == 2:
            self._add('profile', 'Profile', 'gaussian', kind='combo', choices=('gaussian', 'lorentzian', 'pseudovoigt'))
        self._add('use_uncertainties', 'Use admitted uncertainties', False, kind='bool')
        self.advanced = AdvancedDialog(self, title='Region fit component initials')
        self.seed_rows = []
        for component in range(3):
            rows = tuple(FormRow(label=label, path=('region', f'{component}_{name}'), value='')
                         for name, label in (('center', 'Native center'), ('fwhm', 'Initial FWHM'), ('height', 'Initial height')))
            self.advanced.add_group(f'Component p{component+1}', rows)
            self.seed_rows.append(rows)
            for row in rows:
                row.valueChanged.connect(self._edited)
                row.draftChanged.connect(self._draft_changed)
        self.fit_button = self._action('Fit', self.fit)
        self._action('Advanced…', self.advanced.show, ControlAction.ADVANCED_PROCESSING)
        self._action('Cancel', self.cancel)
        self.table = ResultsTable(('Parameter', 'Value', 'StdErr', 'Unit'), max_height=220)
        side.addWidget(self.table)
        self.trend = TrendView(order=True, x_label='Result', x_name='result')
        side.addWidget(self.trend)
        self.trend.save.clicked.disconnect()
        self.trend.save.clicked.connect(self._choose_csv)
        self._payload = self._snapshot = self._outcome = None
        self._name, self._generation = '', 0
        self._log_scale, self._color_map = False, 'Default'
        self._display_index = -1
        self._reports, self._contexts = {}, {}
        self.source_root = None
        self._requests = LatestRequestOwner(_region_job, self._done,
            same=lambda request, other: request.matches(other), parent=self)
        self.export_actions = FitExportActions(self, tool=f'region{ndim}d', snapshot=self.current,
                                               settings=self.export_settings, export=self.export_to)
        self.resize(1200, 780)

    def _add(self, name, label, value, **kwargs):
        row = FormRow(label=label, path=('region', name), value=value, **kwargs)
        self.card.add_row(row)
        self.rows[name] = row
        row.valueChanged.connect(self._edited)
        row.draftChanged.connect(self._draft_changed)
        return row

    def _action(self, label, callback, action=ControlAction.PREVIEW):
        button = ActionButton(ControlActionSpec(action, label, SectionId.PROCESSING))
        button.clicked.connect(callback)
        self.card.add_row(button)
        return button

    @property
    def outcome(self):
        return self._outcome

    @property
    def computing(self):
        return self._requests.computing

    def active(self):
        return self._requests.running is not None or self.computing

    def show_result(self, name, payload):
        if type(payload) is not AnalysisArtifactPayload or payload.kind is not AnalysisArtifactKind.RSM:
            raise TypeError('region fitting requires an admitted RSM AnalysisArtifactPayload')
        if payload is self._payload and str(name) == self._name:
            return
        self.cancel()
        self._generation += 1
        self._payload, self._name = payload, str(name)
        self.setWindowTitle(f'{self.ndim}-D Region Fit — {Path(str(name)).name}')
        names = tuple(axis for axis, _values in payload.axes)
        editor = self.rows['plane'].editor
        editor.blockSignals(True)
        editor.clear()
        editor.addItems(names)
        editor.setCurrentText(names[-1])
        editor.blockSignals(False)
        if self.ndim == 2:
            self.rows['index'].editor.setText(str(payload.intensity.shape[-1]//2))
        for row in self.range_rows.values():
            self.card.body_layout.removeWidget(row)
            row.deleteLater()
        self.range_rows.clear()
        for axis, coords in payload.axes:
            row = RangeRow(label=axis, low={'path': ('box', axis, 'low'), 'value': float(np.min(coords))},
                           high={'path': ('box', axis, 'high'), 'value': float(np.max(coords))})
            self.card.add_row(row)
            row.valueChanged.connect(self._edited)
            row.draftChanged.connect(self._draft_changed)
            self.range_rows[axis] = row
        selected = names if self.ndim == 3 else names[:-1]
        coordinates = dict(payload.axes)
        default = ' '.join(str(float((coordinates[axis][0]+coordinates[axis][-1])/2)) for axis in selected)
        for rows in self.seed_rows:
            rows[0].editor.setPlaceholderText(default)
        self._display_index = -1
        self._refresh()

    def selection(self):
        bounds = tuple((name, tuple(float(editor.text()) for editor in row.editors))
                       for name, row in self.range_rows.items())
        for _axis, (low, high) in bounds:
            if not np.isfinite((low, high)).all() or not low < high:
                raise ValueError('FIT_REGION_BOUNDS_INVALID: enter two increasing finite bounds')
        selection = (('bounds', bounds),)
        if self.ndim == 2:
            selection += (('plane', self.rows['plane'].current_value()),
                          ('index', int(self.rows['index'].editor.text())))
        return selection

    def fit_settings(self):
        settings = {'background': self.rows['background'].current_value(),
                    'use_uncertainties': self.rows['use_uncertainties'].current_value()}
        if self.ndim == 2:
            settings['profile'] = self.rows['profile'].current_value()
        return settings

    def current(self):
        if self._payload is None:
            return None
        request = self._request()
        return FitTarget(request.key.result_identity, lambda: rsm_region_snapshot(request.payload, request.key,
            bounds=request.bounds, plane=request.plane if self.ndim == 2 else None,
            index=request.index if self.ndim == 2 else None), reference=_reference(request))

    def export_settings(self):
        return _settings(self._request())

    def export_to(self, path):
        """Queue current input/settings publication on the same worker, before a fit is required."""
        try:
            request = replace(self._request(), snapshot=None, outcome=None, destination=Path(path))
        except (ValueError, TypeError) as error:
            self.card.set_status_text(str(error))
            return
        self.card.set_status_text('Exporting current region…')
        self._requests.ask(request)

    def export_csv(self, path):
        table = outcomes_table(tuple(self._reports.values()), contexts=self._contexts, model=f'region{self.ndim}d')
        Path(path).write_text(table_csv(table), encoding='utf-8')

    def _choose_csv(self):
        from xdart.gui.widgets.file_dialogs import browse
        picks = browse(self, 'Save region fit results', save=True,
                       current='' if self._payload is None else self._payload.inspection.path,
                       default_name='region_fit.csv', file_filter='CSV (*.csv)')
        if picks:
            try:
                self.export_csv(picks[0])
                self.card.set_status_text(f'Saved {picks[0]}')
            except (OSError, ValueError) as error:
                self.card.set_status_text(str(error))

    def _request(self, solve=False):
        if self._payload is None:
            raise ValueError('FIT_INPUT_UNAVAILABLE: select an RSM result')
        selection = self.selection()
        settings = self.fit_settings()
        plane = self.rows['plane'].current_value()
        axes = tuple((name, values) for name, values in self._payload.axes if self.ndim == 3 or name != plane)
        count = int(self.rows['components'].editor.text())
        if not 1 <= count <= 3:
            raise ValueError('FIT_COMPONENT_LIMIT: use one to three components')
        bounds = dict(dict(selection)['bounds'])
        if np.prod([np.count_nonzero((values >= bounds[name][0]) & (values <= bounds[name][1]))
                    for name, values in axes]) > MAX_POINTS:
            raise ValueError('FIT_REGION_POINT_LIMIT: narrow the native ranges')
        seeds = []
        for i, rows in enumerate(self.seed_rows[:count]):
            center = rows[0].editor.text().strip()
            if i and not center:
                raise ValueError(f'FIT_SEED_REQUIRED: enter the native center of component p{i+1}')
            center = center or ' '.join(str(float((a[0]+a[-1])/2)) for _n, a in axes)
            center = self._numbers(center, len(axes))
            fwhm = self._numbers(rows[1].editor.text(), len(axes)) if rows[1].editor.text().strip() else None
            height = float(rows[2].editor.text()) if rows[2].editor.text().strip() else None
            seeds.append(PeakSeed(f'p{i+1}', center, fwhm, height))
        seeds = tuple(seeds)
        frozen_settings = tuple(sorted(settings.items()))
        key_settings = frozen_settings + (('seeds', tuple((s.id, s.center, s.fwhm, s.height) for s in seeds)),)
        path = self._name if Path(self._name).is_absolute() else self._payload.inspection.path
        key = FitInputKey(self._name, path, self._payload.result_fingerprint,
                          self._generation, selection=selection, settings=key_settings)
        return _RegionRequest(self._payload, key, self.ndim, tuple(bounds.items()), plane,
            int(self.rows['index'].editor.text()) if self.ndim == 2 else self._display_index,
            seeds, frozen_settings, solve, self._snapshot if not solve else None, self._outcome if not solve else None,
            source_root=self.source_root)

    @staticmethod
    def _numbers(text, count):
        values = tuple(float(part) for part in str(text).replace(',', ' ').split())
        if len(values) != count or not np.isfinite(values).all():
            raise ValueError(f'FIT_SEED_INVALID: enter {count} finite native coordinates')
        return values

    def _draft_changed(self, *_args):
        self.cancel()

    def _edited(self, path, _value):
        # 3-D plane selection is presentation; 2-D plane/index selects fitted data.
        if tuple(path) == ('region', 'plane') and self.ndim == 3:
            self._display_index = -1
            self._refresh()
            return
        self.cancel()
        self._display_index = -1
        self._refresh()

    def fit(self):
        self._refresh(solve=True)

    def _refresh(self, solve=False):
        try:
            request = self._request(solve)
        except (ValueError, TypeError) as error:
            self.cancel()
            self.card.set_status_text(str(error))
            return
        if self._requests.ask(request):
            self.card.set_status_text('Fitting…' if solve else 'Preparing native comparison…')

    def _done(self, request, terminal):
        if terminal.status is not OperationTerminalStatus.RETURNED:
            self.card.set_status_text(f'{terminal.failure_type}: {terminal.failure_message}')
            return False
        if request.destination is not None:
            self.card.set_status_text(f'Exported {terminal.payload}')
            return False  # exporting did not replace the current comparison
        snapshot, outcome, analysis, panels = terminal.payload
        self._snapshot, self._outcome = snapshot, outcome
        if self.ndim == 3:
            self._display_index = panels[0].index
        self.view.reconcile_keyed(KeyedPanelsProjection((_KEYS,), panels,
            title=Path(self._name).name, log_scale=self._log_scale, color_map=self._color_map, share_axis=True))
        if self.ndim == 2:
            x, y = snapshot.axes
            bounds = dict(request.bounds)
            self.view.set_keyed_region(_KEYS[0], RectRegion(bounds[x.name], bounds[y.name],
                                                           (x.name, x.unit), (y.name, y.unit)))
        if analysis is not None:
            self._reports[self._name] = replace(analysis, result=None, overlay=None)
            self._contexts[self._name] = _TableContext(snapshot.key,
                tuple(_AxisUnit(axis.name, axis.unit) for axis in snapshot.axes))
            self.table.show_rows(('Parameter', 'Value', 'StdErr', 'Unit'), tuple(
                (name, value, analysis.parameter_errors.get(name), analysis.parameter_units.get(name, ''))
                for name, value in analysis.parameter_values.items()))
            self.trend.set_row(self._name, None, analysis.params)
        self.card.set_status_text(outcome.physical_reason or outcome.diagnostics.message if outcome else 'Ready to fit or export')
        return True

    def _view_command(self, command):
        if command.kind is ShellCommandKind.SET_PANEL_INDEX and self.ndim == 3:
            self._display_index = int(command.value)
            self._refresh()
        elif command.kind is ShellCommandKind.SET_LOG_SCALE:
            self._log_scale = bool(command.value)
            self._redraw()
        elif command.kind is ShellCommandKind.SET_COLOR_MAP:
            self._color_map = str(command.value)
            self._redraw()

    def _redraw(self):
        from dataclasses import replace
        state = self.view.keyed_projection
        if state is not None:
            self.view.reconcile_keyed(replace(state, log_scale=self._log_scale, color_map=self._color_map))

    def _region_changed(self, _key, region):
        if region is None or self.ndim != 2:
            return
        for name, bounds in ((region.x_axis[0], region.x), (region.y_axis[0], region.y)):
            for editor, value in zip(self.range_rows[name].editors, bounds):
                editor.setText(str(value))
        self.cancel()
        self._refresh()

    def cancel(self):
        self._requests.release()
        self._snapshot = self._outcome = None
        self.table.clear_rows()
        self.view.clear_keyed()

    def release(self):
        self.cancel()
        self._payload = None
        self.advanced.hide()
        self._generation += 1

    def shutdown(self):
        self.release()
        return self._requests.shutdown()

    def hideEvent(self, event):
        if not event.spontaneous():
            self.release()
        super().hideEvent(event)


__all__ = ['RegionFitWindow']
