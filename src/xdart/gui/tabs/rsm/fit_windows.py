"""RSM native axis lineouts in the same Peak Fit window used by Int and Stitch."""
from pathlib import Path

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.analysis.snapshot_fit_host import FitTarget, SnapshotFitHost
from xdart.gui.pages.values import CloseReceipt, PageCleanup
from xdart.gui.widgets.controls_panel import FormRow, SectionCard
from xrd_tools.analysis.fit_snapshot import FitInputKey
from xrd_tools.analysis.fit_sources import rsm_lineout_snapshot
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, AnalysisArtifactPayload
from xrd_tools.notebooks.inputs import SourceReference


class RSMLineoutWindow(QtWidgets.QDialog):
    """Select a native axis, through point and full bandwidths; extraction/fit runs on the shared host worker.

    Batch availability is chosen before the first ``show_result``; a later
    ``configure_batch`` updates its provider and project root. The provider
    receives immutable selection pairs and returns cheap FitTargets.
    Batch is unavailable until the page explicitly supplies that capability.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setObjectName('rsmLineoutWindow')
        self.setWindowTitle('RSM Lineout Fit')
        self._layout = QtWidgets.QVBoxLayout(self)
        self.card = SectionCard('Recorded volume — native lineout, voxel mean over ± full width / 2', collapsible=False)
        self.card.setToolTip('Uses the recorded linear volume; the RSM main view\'s Set BG is not applied.')
        self._layout.addWidget(self.card)
        self.rows = {}
        self._payload, self._name, self._generation = None, '', 0
        self.host = self.fit_dialog = None
        self._batch_provider, self._source_root = None, None
        self.resize(650, 900)

    def configure_batch(self, provider, source_root=None):
        if self.host is not None and (provider is None) != (self.host.batch is None):
            raise RuntimeError('Batch availability is chosen before the first show_result')
        changed = source_root != self._source_root
        self._batch_provider, self._source_root = provider, source_root
        if self.host is not None:
            self.host.batch = None if provider is None else self.batch
            self.host.source_root = source_root
            self.fit_dialog.export_actions.source_root = source_root
            if changed:
                self._generation += 1
                self.host.invalidate()
                if self._payload is not None:
                    self.host.reload()

    def _mount_host(self):
        if self.host is not None:
            return
        self.host = SnapshotFitHost('peak', self, current=self.current, scope=self.scope,
            batch=None if self._batch_provider is None else self.batch, source_root=self._source_root)
        self.fit_dialog = self.host.dialog
        self.fit_dialog.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.fit_dialog.finished.connect(self.close)
        self._layout.addWidget(self.fit_dialog, 1)

    def show_result(self, name, payload):
        if type(payload) is not AnalysisArtifactPayload or payload.kind is not AnalysisArtifactKind.RSM:
            raise TypeError('lineout fitting requires an admitted RSM AnalysisArtifactPayload')
        if payload is self._payload and str(name) == self._name:
            return
        if self.host is not None:
            self.host.invalidate()
        self._payload, self._name = payload, str(name)
        self._generation += 1
        self.setWindowTitle(f'RSM Lineout Fit — {Path(str(name)).name}')
        names = tuple(name for name, _axis in payload.axes)
        if tuple(name[7:] for name in self.rows if name.startswith('center_')) != names:
            self.card.clear_rows()
            self.rows = {'axis': FormRow(label='Along axis', path=('lineout', 'axis'),
                                        value=names[0], kind='combo', choices=names)}
            for axis in names:
                self.rows[f'center_{axis}'] = FormRow(label=f'Through {axis}', path=('lineout', f'center_{axis}'), value='')
                self.rows[f'width_{axis}'] = FormRow(label=f'Full width {axis}', path=('lineout', f'width_{axis}'), value='0')
            for row in self.rows.values():
                self.card.add_row(row)
                row.valueChanged.connect(self._edited)
                row.draftChanged.connect(self._draft_changed)
        for axis, values in payload.axes:
            self.rows[f'center_{axis}'].editor.setPlaceholderText(f'Auto: {(values[0]+values[-1])/2:.6g}')
        self._mount_host()
        self.host.reload()

    def selection(self):
        if self._payload is None:
            raise ValueError('FIT_INPUT_UNAVAILABLE: select an RSM result')
        centre, width = [], []
        for name, axis in self._payload.axes:
            text = self.rows[f'center_{name}'].editor.text().strip()
            centre.append(float(text) if text else float((axis[0]+axis[-1])/2))
            width.append(float(self.rows[f'width_{name}'].editor.text()))
        if not np.isfinite((*centre, *width)).all() or any(value < 0 for value in width):
            raise ValueError('FIT_LINEOUT_INVALID: finite native points and nonnegative full bandwidths required')
        return (('axis', self.rows['axis'].current_value()), ('centre', tuple(centre)),
                ('width', tuple(width)), ('meaning', 'axis_lineout'))

    def current(self):
        if self._payload is None:
            return None
        payload, selection = self._payload, self.selection()
        path = self._name if Path(self._name).is_absolute() else payload.inspection.path
        key = FitInputKey(self._name, path, payload.result_fingerprint,
                          self._generation, selection=selection)
        values = dict(selection)
        reference = SourceReference(path, 'rsm_lineout', self._name,
            payload.result_fingerprint, payload.inspection.request_fingerprint,
            selection=selection[:3], original_path=path)
        return FitTarget(self._name, lambda: rsm_lineout_snapshot(payload, key, axis=values['axis'],
                         centre=values['centre'], width=values['width']), reference=reference)

    def scope(self):
        return None if self._payload is None else (
            self._name, self._payload.result_fingerprint, self.selection(), self._generation,
            tuple(target.label for target in self.batch()))

    def batch(self):
        return () if self._batch_provider is None else tuple(self._batch_provider(self.selection()))

    def _edited(self, *_args):
        if self.host is not None:
            self.host.reload()

    def _draft_changed(self, *_args):
        if self.host is not None:
            self.host.invalidate()

    @property
    def computing(self):
        return self.host is not None and self.host.computing

    def active(self):
        return self.host is not None and (self.computing or self.host.owner.running is not None)

    def release(self):
        self._payload = None
        self._generation += 1
        if self.host is not None:
            self.host.invalidate()

    def shutdown(self):
        self.release()
        return CloseReceipt(PageCleanup.CLEAN) if self.host is None else self.host.shutdown()

    def hideEvent(self, event):
        if not event.spontaneous():
            self.release()
        super().hideEvent(event)


__all__ = ['RSMLineoutWindow']
