"""Shared fit-window export actions over the Qt-free notebook recipe owner."""
from dataclasses import replace
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets
from xrd_tools.analysis.fit_snapshot import FitAxis, FitInputKey, FitInputSnapshot
from xdart.gui.widgets import file_dialogs


def plain_settings(values):
    """Detach current controls into the snapshot's plain, comparable settings."""
    from xrd_tools.analysis.display_fit_operations import DisplayedTraceInput
    from xrd_tools.analysis.phase_references import PhaseEntry
    def plain(value):
        if isinstance(value, PhaseEntry):
            # The library ID pins CIF bytes; manual cells remain part of the
            # request identity. This is a key projection, not a recipe format.
            return ('phase', value.id, value.name, value.source_kind, plain(value.lattice))
        if isinstance(value, DisplayedTraceInput):
            # A captured substrate (PHASE-ANALYSIS-128) by its trace identity, never its arrays.
            return ('template', value.trace_fingerprint, value.label, value.axis_unit)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return tuple(plain(item) for item in value)
        if isinstance(value, dict):
            return tuple(sorted((str(k), plain(v)) for k, v in value.items()))
        return value
    return tuple(sorted((str(name), plain(value)) for name, value in values.items()))


def write_fit_notebook(path, snapshot, tool, settings, *, batch=(), batch_inputs=(),
                       data_root=None, reciprocal_transform=None, reference=None, trend_column=""):
    """Write the detached current request, including lossless derived values, before a fit is needed."""
    from xrd_tools.notebooks import FitNotebookRecipe, capture_snapshot, render_notebook
    path = Path(path)
    format = 'jupyter' if path.suffix == '.ipynb' else 'marimo' if path.suffix == '.py' else None
    if format is None:
        raise ValueError('NOTEBOOK_FORMAT_REQUIRED: choose .ipynb or .py')
    settings = dict(settings)
    transform = None if reciprocal_transform is None else tuple(tuple(float(v) for v in row)
                                                               for row in reciprocal_transform)
    snapshot = replace(snapshot, key=replace(snapshot.key, settings=plain_settings(settings)))
    captured = None if reference is not None else capture_snapshot(snapshot, path.name + '.input.npz')
    captures, references = ([] if captured is None else [captured]), list(batch)
    charged = snapshot.storage_bytes
    for item in batch_inputs:
        if isinstance(item, FitInputSnapshot):
            charged += item.storage_bytes
            if charged > 32 << 20:
                raise ValueError('INPUT_LIMIT: captured batch exceeds 32 MiB; export a smaller selection')
            part = capture_snapshot(item, f'{path.name}.batch-{len(references) + 1}.input.npz')
            captures.append(part)
            references.append(part.reference)
        else:
            references.append(item)
    recipe = FitNotebookRecipe(tool=tool, input=reference if captured is None else captured.reference,
                               settings=settings, batch=tuple(references),
                               title=snapshot.key.result_identity, reciprocal_transform=transform,
                               trend_column=trend_column,
                               orthonormal=all(a.unit in {'A^-1', 'Å⁻¹', 'q_A^-1'} for a in snapshot.axes))
    document = render_notebook(recipe, format=format, bundle_root=str(path.parent),
                               data_root=None if data_root is None else str(data_root))
    from xrd_tools.io.export import write_exports_atomically
    parts = [(path.parent / item.reference.path, item.content) for item in captures]
    write_exports_atomically((*parts, (path, document)))
    return path


class FitExportActions(QtCore.QObject):
    """The same two export buttons in every fit window; destination uses shared Browse."""
    def __init__(self, parent, *, tool, snapshot, settings, export=None):
        super().__init__(parent)
        self.tool, self.snapshot, self.settings = tool, snapshot, settings
        self.export = export
        self.source_root = None
        self._unavailable_reason = ''
        row = QtWidgets.QHBoxLayout()
        self.buttons = {}
        for format, label in (('jupyter', 'Export Jupyter…'), ('marimo', 'Export marimo…')):
            button = QtWidgets.QPushButton(label, parent)
            button.setToolTip(f"Export the current fitting workflow as a {format} notebook.")
            button.clicked.connect(lambda _checked=False, format=format: self.choose(format))
            row.addWidget(button)
            self.buttons[format] = button
        parent.layout().addLayout(row)

    def set_available(self, available, reason=''):
        """Explain an unsupported workflow through the shared export buttons."""
        self._unavailable_reason = '' if available else str(reason)
        for format, button in self.buttons.items():
            button.setEnabled(bool(available))
            button.setToolTip(self._unavailable_reason or
                             f"Export the current fitting workflow as a {format} notebook.")

    def choose(self, format):
        extension = '.ipynb' if format == 'jupyter' else '.py'
        try:
            if self._unavailable_reason:
                raise ValueError(self._unavailable_reason)
            snapshot = self.snapshot()
            if snapshot is None:
                raise ValueError('FIT_INPUT_REQUIRED: load a scientific input first')
            path = getattr(getattr(snapshot, 'key', None), 'path', '')
            reference = getattr(snapshot, 'reference', None)
            if not path and reference is not None:
                path = reference.original_path or reference.path
            picks = file_dialogs.browse(self.parent(), 'Export fitting workflow', save=True, current=path,
                fallback=str(self.source_root or ''),
                default_name=file_dialogs.export_name(snapshot.key.result_identity, self.tool, extension),
                file_filter=f'Notebook (*{extension})')
            if not picks:
                return
            destination = Path(picks[0]).with_suffix(extension)
            if self.export is not None:
                self.export(destination)
            else:
                write_fit_notebook(destination, snapshot, self.tool, self.settings())
                self.parent().status.setText(f'Exported {destination}')
        except Exception as error:
            self.parent().status.setText(str(error))


def snapshot_from_trace(trace, *, wavelength=None, settings=(), meaning='radial'):
    receipt = trace.receipt
    key = FitInputKey(trace.title, str(receipt.artifact), receipt.run_fingerprint or trace.trace_fingerprint,
                      receipt.run_generation, selection=(('meaning', meaning), ('frame', receipt.frame_label)),
                      settings=settings)
    return FitInputSnapshot(key, (FitAxis(trace.label, trace.axis_unit, trace.axis),), trace.intensity,
                            wavelength_angstrom=wavelength, sigma=trace.sigma)
