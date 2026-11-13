"""Int's shared editor and explicit confirmation for consecutive point images."""
import os
from dataclasses import replace
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.advanced_editor import AdvancedDialog
from xdart.gui.widgets.controls_panel import FormRow, SubsectionCard
from xrd_tools.core.scan import SourceSpec
from xrd_tools.sources.point_images import PointImagePolicy

_OPERATIONS = {'Average': 'average', 'Difference (2 − 1)': 'difference', 'Select one': 'select'}

#: An unanswered grouping question confirms what it shows after this many seconds (NEXT-1210-RESPONSIVENESS C-13; the
#: maintainer, 2026-11-13: "put a wait timer of 30s and if there's no response, go ahead and assume yes as the answer").
AUTO_CONFIRM_SECONDS = 30


def point_source_key(spec):
    return str(spec.uri), str(spec.options.get('scan') or spec.entry or 'entry')


def grouping_file(detection):
    """The file a remembered grouping is kept for (C-13): an image series' SPEC file, else the source's own file, as
    written (shown and saved so); ``_file_key`` matches it."""
    return os.path.abspath(str(detection.metadata_path or detection.source_spec.uri))


def _file_key(path):
    """A remembered file's match key: normalized (Windows: case and separators), never shown."""
    return os.path.normcase(os.path.abspath(str(path)))


_REMEMBERED_TIP = ('Files whose new scans are grouped without asking, each with the grouping it applies (ticked '
                   '"Apply to all scans in this file" when a grouping was confirmed).')


def _for_count(policy, count):
    """A confirmed grouping re-applied to a scan of ``count`` images per point (``compatible_with`` checked)."""
    return replace(policy, images_per_point=count,
                   image_index=policy.image_index if policy.operation == 'select' else 1)


def _grouping_name(policy):
    name = next(name for name, code in _OPERATIONS.items() if code == policy.operation)
    if policy.operation == 'select':
        return f'{name} (image {policy.image_index})'
    return f'{name}, reversed' if policy.reverse else name


def _row(label, value, *, kind='text', choices=(), tip='', minimum=None, maximum=None):
    row = FormRow(label=label, path=('Point images', label), value=value, kind=kind,
                  choices=choices, minimum=minimum, maximum=maximum)
    row.set_purpose_tooltip(tip)
    return row


class PointImagesDialog(AdvancedDialog):
    """Confirmation uses the same shell and form rows as Int's Advanced editor."""

    def __init__(self, detection, parent=None, policy=None, *, countdown=True):
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
        self.remember_file = _row('Apply to all scans in this file, including new ones', False, kind='bool',
            tip=f'Saved with these settings for {Path(grouping_file(detection)).name}: a later scan of this file '
                'whose image count fits is grouped this way without asking. Images per point lists the remembered '
                'files; untick one there to be asked again.')
        self.add_section(self.remember_file)
        self.operation.valueChanged.connect(lambda *_: self._sync())
        self.index.editor.textChanged.connect(lambda *_: self._sync())
        if policy is not None and policy.compatible_with(count):
            self.operation.editor.setCurrentText(next(name for name, code in _OPERATIONS.items()
                                                       if code == policy.operation))
            self.index.editor.setText(str(policy.image_index))
            self.reverse.editor.setChecked(policy.reverse)
        self._sync()
        # C-13: left unanswered, the question confirms what it shows (for this run's remaining scans too); any change
        # in it stops the countdown.
        self.auto_confirmed = False
        self._seconds = AUTO_CONFIRM_SECONDS
        self._counting = bool(countdown)
        self._countdown = QtCore.QTimer(self)
        self._countdown.setInterval(1000)
        self._countdown.timeout.connect(self._tick)
        for row in (self.operation, self.index, self.reverse, self.apply_remaining, self.remember_file):
            row.valueChanged.connect(self._stop_countdown)
        self.index.editor.textEdited.connect(self._stop_countdown)
        self._label_confirm()
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setToolTip(
            'Confirm consecutive image membership and the displayed operation for this scan.'
            + (f' Left unanswered, it confirms itself after {AUTO_CONFIRM_SECONDS} s, for the remaining scans of '
               'this run too; any change here stops the countdown.' if countdown else ''))
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setToolTip(
            'Leave this scan unprocessed; no grouping is assumed.')

    def showEvent(self, event):
        super().showEvent(event)
        if self._counting and not self._countdown.isActive():
            self._countdown.start()

    def _label_confirm(self):
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText(
            f'Confirm grouping ({self._seconds} s)' if self._counting else 'Confirm grouping')

    def _tick(self):
        self._seconds -= 1
        if self._seconds > 0:
            self._label_confirm()
            return
        self._countdown.stop()
        if self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).isEnabled():
            self.auto_confirmed = True
            self.accept()

    def _stop_countdown(self, *_):
        if self._counting:
            self._counting = False
            self._countdown.stop()
            self._label_confirm()

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
        self._file_defaults = {}  # C-13: _file_key -> (the file as written, the grouping its new scans get)
        self._remembered_shown = []
        self._answered_silently = set()  # C-13: the scans answered without a question, as found
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
        self.remembered = _row('Remembered file', 'None', kind='combo', choices=('None',), tip=_REMEMBERED_TIP)
        self.remember = _row('Group its new scans without asking', False, kind='bool',
                             tip='Ticked, a later scan of the file shown above whose image count fits is grouped '
                                 'that way without asking (saved with these settings). Untick to be asked again '
                                 'for that file.')
        self.count.editor.setReadOnly(True)
        for row in (self.scan, self.count, self.operation, self.index, self.reverse, self.remembered, self.remember):
            self.add_row(row)
        self.scan.valueChanged.connect(lambda *_: self._render())
        for row in (self.operation, self.index, self.reverse):
            row.valueChanged.connect(self._edited)
        self.remember.valueChanged.connect(self._forget)
        self._render()
        self._catalog_remembered()

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

    def _catalog_remembered(self):
        self._rendering = True
        try:
            self._remembered_shown = list(self._file_defaults)
            files = [file for file, _policy in self._file_defaults.values()]
            self.remembered.editor.clear()
            self.remembered.editor.addItems([f'{Path(file).name} · {_grouping_name(policy)}'
                                             for file, policy in self._file_defaults.values()] or ['None'])
            self.remembered.editor.setToolTip('\n\n'.join((_REMEMBERED_TIP, '\n'.join(files)))
                                              if files else _REMEMBERED_TIP)
            self.remember.editor.setChecked(bool(self._remembered_shown))
            self.remember.setEnabled(bool(self._remembered_shown))
        finally:
            self._rendering = False

    def _forget(self, *_):
        """Unticked: the shown file's new scans are asked about again (its confirmed scans keep their grouping)."""
        if self._rendering or self.remember.current_value():
            return
        index = self.remembered.editor.currentIndex()
        if 0 <= index < len(self._remembered_shown):
            self._file_defaults.pop(self._remembered_shown[index], None)
        self._catalog_remembered()

    def options(self, spec):
        policy = self._choices.get(point_source_key(spec))
        return spec if policy is None else replace(spec, options={**spec.options, 'point_images': policy.to_dict()})

    def confirm(self, detections, parent=None):
        for detection in detections:
            count = detection.images_per_point
            key, file = point_source_key(detection.source_spec), grouping_file(detection)
            remembered = self._file_defaults.get(_file_key(file), (None, None))[1]
            # its file's remembered grouping first, then this run's answer for the remaining scans (C-13)
            policy = next((_for_count(candidate, count) for candidate in (remembered, self._remaining)
                           if candidate is not None and candidate.compatible_with(count)), None)
            # The question was the only brake on search -> ask -> search: a scan found again as it was when it was
            # answered without a question is asked about, without a countdown, rather than answered again.
            found = (key, detection.scan_label, count, _file_key(file))
            repeated = policy is not None and found in self._answered_silently
            if policy is None or repeated:
                self._answered_silently.discard(found)
                dialog = PointImagesDialog(detection, parent, self._choices.get(key) or remembered,
                                           countdown=not repeated)
                try:
                    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                        self._remaining = None
                        return False
                    chosen = dialog.policy()
                    answered_all = dialog.apply_remaining.current_value() or dialog.auto_confirmed
                    self._remaining = chosen if answered_all else None
                    if dialog.remember_file.current_value():
                        self._file_defaults[_file_key(file)] = (file, chosen)
                finally:
                    dialog.deleteLater()
                policy = _for_count(chosen, count)
            else:
                self._answered_silently.add(found)
            self._choices[key] = policy
        self._catalog()
        self._catalog_remembered()
        return True

    def finish_run(self):
        self._remaining = None

    def document(self):
        document = {'choices': [{'uri': uri, 'scan': scan, 'policy': policy.to_dict()}
                                for uri, scan, policy in self.choices]}
        if self._file_defaults:  # C-13; a document without a remembered file is unchanged
            document['file_defaults'] = [{'file': file, 'policy': policy.to_dict()}
                                         for file, policy in self._file_defaults.values()]
        return document

    def restore(self, value):
        choices = {}
        for row in value.get('choices', ()):
            policy = PointImagePolicy.from_dict(row['policy'])
            choices[(str(row['uri']), str(row['scan']))] = policy
        defaults = {}
        for row in value.get('file_defaults', ()):
            if type(row.get('file')) is not str:
                raise ValueError('a remembered grouping names its file')
            defaults[_file_key(row['file'])] = (row['file'], PointImagePolicy.from_dict(row['policy']))
        self._choices = choices
        self._file_defaults = defaults
        self._answered_silently.clear()
        self.finish_run()
        self._catalog()
        self._catalog_remembered()

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
        if self._choices or self._file_defaults:
            intent.run_options['point_image_choices'] = self.document()
        if type(intent.source_spec) is SourceSpec:
            intent.source_spec = self.options(intent.source_spec)
        return intent
