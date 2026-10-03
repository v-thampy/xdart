"""Int's shared editor and explicit confirmation for consecutive point images."""
from dataclasses import replace

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.advanced_editor import AdvancedDialog
from xdart.gui.widgets.controls_panel import FormRow, SubsectionCard
from xrd_tools.core.scan import SourceSpec
from xrd_tools.sources.point_images import PointImagePolicy

_OPERATIONS = {'Average': 'average', 'Difference (2 − 1)': 'difference', 'Select one': 'select'}


def point_source_key(spec):
    return str(spec.uri), str(spec.options.get('scan') or spec.entry or 'entry')


def _row(label, value, *, kind='text', choices=(), tip='', minimum=None, maximum=None):
    row = FormRow(label=label, path=('Point images', label), value=value, kind=kind,
                  choices=choices, minimum=minimum, maximum=maximum)
    row.set_purpose_tooltip(tip)
    return row


class PointImagesDialog(AdvancedDialog):
    """Confirmation uses the same shell and form rows as Int's Advanced editor."""

    def __init__(self, detection, parent=None, policy=None):
        super().__init__(parent, title='Multiple images per scan point', accept_buttons=True)
        self.setModal(True)
        self.detection = detection
        count = detection.images_per_point
        card = self.add_group(f'{detection.scan_label}: {detection.image_count} images / '
                              f'{detection.point_count} points = {count} images/pt')
        card.setToolTip('Confirm only when consecutive images belong to the same scan point. '
                        'A count ratio suggests this grouping; it does not prove acquisition order.')
        self.operation = _row('Combine', 'Average', kind='combo',
            choices=tuple(name for name, code in _OPERATIONS.items() if code != 'difference' or count == 2),
            tip='Average each consecutive group, subtract Image 1 from Image 2, or keep one image. '
                'Native invalid/threshold pixels are excluded before combination; differences keep negative values.')
        self.index = _row('Image', 1, kind='int', minimum=1, maximum=count,
            tip='Image to keep within each point, starting at 1. Used only for Select one.')
        self.reverse = _row('Reverse difference', False, kind='bool',
            tip='Use Image 1 minus Image 2 instead of Image 2 minus Image 1; available for two images/pt.')
        for row in (self.operation, self.index, self.reverse):
            card.add_row(row)
        self.apply_remaining = _row('Apply to compatible remaining scans in this run', False, kind='bool',
            tip='Average accepts any group size; Difference requires two; Select requires the chosen image '
                'index. Incompatible scans ask again. This choice is not saved.')
        self.add_section(self.apply_remaining)
        self.operation.valueChanged.connect(lambda *_: self._sync())
        self.index.editor.textChanged.connect(lambda *_: self._sync())
        if policy is not None and policy.compatible_with(count):
            self.operation.editor.setCurrentText(next(name for name, code in _OPERATIONS.items()
                                                       if code == policy.operation))
            self.index.editor.setText(str(policy.image_index))
            self.reverse.editor.setChecked(policy.reverse)
        self._sync()
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText('Confirm grouping')
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setToolTip(
            'Confirm consecutive image membership and the displayed operation for this scan.')
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setToolTip(
            'Leave this scan unprocessed; no grouping is assumed.')

    def _sync(self):
        operation = _OPERATIONS[self.operation.current_value()]
        self.index.setVisible(operation == 'select')
        self.reverse.setVisible(operation == 'difference')
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            operation != 'select' or not self.index.refusal())

    def policy(self):
        operation = _OPERATIONS[self.operation.current_value()]
        return PointImagePolicy(self.detection.images_per_point,
            operation, int(self.index.current_value()) if operation == 'select' else 1,
            bool(self.reverse.current_value()) if operation == 'difference' else False)


class PointImagesSettings(SubsectionCard):
    """Shared Advanced card, with immutable per-source choices and a run-only default."""

    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__('Images per point', parent)
        self.setToolTip('Revisit confirmed consecutive-image grouping for each source scan. '
                        'One logical frame and its metadata represent each scan point.')
        self._choices = {}
        self._shown = []
        self._remaining = None
        self._rendering = False
        self.scan = _row('Scan', 'None confirmed', kind='combo', choices=('None confirmed',),
                         tip='Choose a confirmed scan whose images-per-point settings you want to change.')
        self.count = _row('Images/pt', '', tip='Confirmed number of consecutive images in each scan point.')
        self.operation = _row('Combine', 'Average', kind='combo', choices=tuple(_OPERATIONS),
                              tip='Operation on the confirmed consecutive group. Difference requires two images/pt.')
        self.index = _row('Image', 1, kind='int', minimum=1,
                         tip='The one-based image to keep within each point when Select one is chosen.')
        self.reverse = _row('Reverse difference', False, kind='bool',
                            tip='Use Image 1 minus Image 2 instead of Image 2 minus Image 1.')
        self.count.editor.setReadOnly(True)
        for row in (self.scan, self.count, self.operation, self.index, self.reverse):
            self.add_row(row)
        self.scan.valueChanged.connect(lambda *_: self._render())
        for row in (self.operation, self.index, self.reverse):
            row.valueChanged.connect(self._edited)
        self._render()

    @property
    def choices(self):
        return tuple((uri, scan, policy) for (uri, scan), policy in self._choices.items())

    def _render(self):
        self._rendering = True
        try:
            index = self.scan.editor.currentIndex()
            policy = self._choices[self._shown[index]] if 0 <= index < len(self._shown) else None
            if policy is not None:
                self.count.editor.setText(str(policy.images_per_point))
                self.operation.editor.clear()
                self.operation.editor.addItems([name for name, code in _OPERATIONS.items()
                    if code != 'difference' or policy.images_per_point == 2])
                self.operation.editor.setCurrentText(next(name for name, code in _OPERATIONS.items()
                                                          if code == policy.operation))
                self.index.set_integer_bounds(1, policy.images_per_point)
                self.index.editor.setText(str(policy.image_index))
                self.reverse.editor.setChecked(policy.reverse)
            else:
                self.count.editor.clear()
            for row in (self.operation, self.index, self.reverse):
                row.setEnabled(policy is not None)
            operation = _OPERATIONS[self.operation.current_value()]
            self.index.setVisible(operation == 'select')
            self.reverse.setVisible(operation == 'difference')
        finally:
            self._rendering = False

    def _edited(self, *_):
        if self._rendering or not self._shown:
            return
        key = self._shown[self.scan.editor.currentIndex()]
        old = self._choices[key]
        try:
            operation = _OPERATIONS[self.operation.current_value()]
            self._choices[key] = PointImagePolicy(old.images_per_point,
                operation, int(self.index.current_value()) if operation == 'select' else 1,
                bool(self.reverse.current_value()) if operation == 'difference' else False)
        except (TypeError, ValueError):
            self._render()
            return
        self._render()
        self.changed.emit()

    def _catalog(self):
        self._shown = list(self._choices)
        self.scan.editor.clear()
        self.scan.editor.addItems([f'{uri.rsplit("/", 1)[-1]} · {scan}' for uri, scan in self._shown]
                                  or ['None confirmed'])
        self._render()

    def options(self, spec):
        policy = self._choices.get(point_source_key(spec))
        return spec if policy is None else replace(spec, options={**spec.options, 'point_images': policy.to_dict()})

    def confirm(self, detections, parent=None):
        for detection in detections:
            count = detection.images_per_point
            policy = self._remaining
            if policy is None or not policy.compatible_with(count):
                dialog = PointImagesDialog(detection, parent, self._choices.get(point_source_key(detection.source_spec)))
                try:
                    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                        self._remaining = None
                        return False
                    policy = dialog.policy()
                    self._remaining = policy if dialog.apply_remaining.current_value() else None
                finally:
                    dialog.deleteLater()
            self._choices[point_source_key(detection.source_spec)] = replace(policy, images_per_point=count,
                image_index=policy.image_index if policy.operation == 'select' else 1)
        self._catalog()
        return True

    def finish_run(self):
        self._remaining = None

    def document(self):
        return {'choices': [{'uri': uri, 'scan': scan, 'policy': policy.to_dict()}
                            for uri, scan, policy in self.choices]}

    def restore(self, value):
        choices = {}
        for row in value.get('choices', ()):
            policy = PointImagePolicy.from_dict(row['policy'])
            choices[(str(row['uri']), str(row['scan']))] = policy
        self._choices = choices
        self.finish_run()
        self._catalog()

    def restore_intent(self, intent):
        """Int profiles keep all source choices in run options, alongside its current source policy."""
        document = intent.run_options.get('point_image_choices', {'choices': []})
        self.restore(document)
        spec = intent.source_spec
        if type(spec) is SourceSpec and 'point_images' in spec.options:
            self._choices[point_source_key(spec)] = PointImagePolicy.from_dict(spec.options['point_images'])
            self._catalog()

    def apply_intent(self, intent):
        """Apply only confirmed source choices; the run-only checkbox has no persisted value."""
        if self._choices:
            intent.run_options['point_image_choices'] = self.document()
        if type(intent.source_spec) is SourceSpec:
            intent.source_spec = self.options(intent.source_spec)
        return intent
