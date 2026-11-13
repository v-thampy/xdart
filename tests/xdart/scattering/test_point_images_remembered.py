"""NEXT-1210-RESPONSIVENESS C-13: a confirmed grouping can be remembered for its file, and the question confirms itself.

The maintainer, 2026-11-13, on the images-per-point question that holds an All Preview (verbatim): "can we add an
option of "Apply to all" so it's not waiting every time"; his pick "Remember per SPEC file (Recommended)"; then "put a
wait timer of 30s and if there's no response, go ahead and assume yes as the answer", his pick "Timer on the existing
question (Recommended)".

Int's shared owner (``PointImagesSettings``, mounted by Int, Stitch, RSM and the calibration window) answers a later
scan of a remembered file whose count fits without asking, the file's own grouping before a run-only answer given for
another file; it saves the remembered files with its document, adds nothing to a document without them, and its card
forgets one. An Int image series' detection names the SPEC file its scan came from, so Int remembers per SPEC file too.
Left alone, the question confirms the grouping it shows after its countdown, for the rest of that run's scans; a change
in it stops the countdown.
"""
import os

import pytest
from pyqtgraph.Qt import QtCore, QtWidgets


@pytest.fixture
def qapp(_xdart_qt_harness):
    return _xdart_qt_harness.app


def _detected(root, file, scan, count):
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.sources.point_images import PointImageDetection
    spec = SourceSpec(root / file, SourceKind.SPEC, options={'scan': scan})
    return PointImageDetection(scan, 2, 2 * count, count, spec)


class _Answers:
    """Answers each grouping question as it opens: (operation, remaining, remember) per question in turn."""

    def __init__(self, qapp, answers):
        self.qapp, self.answers, self.seen, self.errors = qapp, list(answers), [], []
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._answer)
        self.timer.start(5)

    def _answer(self):
        from xdart.gui.tabs.scattering.point_images_editor import PointImagesDialog
        dialog = self.qapp.activeModalWidget()
        if not isinstance(dialog, PointImagesDialog) or dialog in self.seen:
            return
        self.seen.append(dialog)
        try:
            operation, remaining, remember = self.answers[len(self.seen) - 1]
            dialog.operation.editor.setCurrentText(operation)
            dialog.operation.valueChanged.emit(dialog.operation.path, operation)
            dialog.apply_remaining.editor.setChecked(remaining)
            row = getattr(dialog, 'remember_file', None)
            if row is None:
                raise AssertionError('the question has no "remember for this file" choice')
            row.editor.setChecked(remember)
            dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()
        except Exception as error:  # noqa: BLE001 - reported by the row
            self.errors.append(error)
            dialog.reject()

    def stop(self):
        self.timer.stop()


def _policies(settings):
    return {scan: (policy.operation, policy.images_per_point) for _uri, scan, policy in settings.choices}


def test_a_remembered_file_answers_its_new_scans_and_the_card_forgets_it(qapp, tmp_path):
    from xdart.gui.tabs.scattering.point_images_editor import PointImagesSettings
    settings = PointImagesSettings()
    answers = _Answers(qapp, [('Average', False, True),         # file A: remembered
                              ('Select one', True, False),      # file D: this run only
                              ('Difference (2 − 1)', False, True),  # file C: remembered, two images only
                              ('Average', False, False)])       # file C again: three images, so asked
    try:
        assert settings.confirm((_detected(tmp_path, 'A', '1.1', 2),)), answers.errors
        settings.finish_run()
        assert settings.confirm((_detected(tmp_path, 'A', '2.1', 2), _detected(tmp_path, 'A', '3.1', 3)))
        assert len(answers.seen) == 1, 'a new scan of a remembered file is answered without asking'
        assert _policies(settings)['3.1'] == ('average', 3)
        settings.finish_run()
        # the file's own grouping wins over a run-only answer given for another file in the same run
        assert settings.confirm((_detected(tmp_path, 'D', '1.1', 2), _detected(tmp_path, 'A', '4.1', 2)))
        assert len(answers.seen) == 2 and _policies(settings)['4.1'] == ('average', 2)
        settings.finish_run()
        assert settings.confirm((_detected(tmp_path, 'C', '1.1', 2),))
        settings.finish_run()
        assert settings.confirm((_detected(tmp_path, 'C', '2.1', 3),))
        assert len(answers.seen) == 4, 'a remembered grouping that cannot apply asks again'
        assert not answers.errors, answers.errors
        document = settings.document()
        assert 'file_defaults' in document
        restored = PointImagesSettings()
        restored.restore(document)
        assert restored.document() == document
        assert restored.confirm((_detected(tmp_path, 'A', '5.1', 2),))
        assert len(answers.seen) == 4, 'the remembered file survives its settings'
        # the card lists the remembered files; unticking forgets the shown one
        assert restored.remembered.editor.count() == 2, [restored.remembered.editor.itemText(i) for i in range(2)]
        assert restored.remembered.editor.currentText().startswith('A'), restored.remembered.editor.currentText()
        assert restored.remember.current_value() and restored.remember.toolTip() and restored.remembered.toolTip()
        restored.remember.editor.setChecked(False)
        assert restored.remembered.editor.count() == 1 and restored.remembered.editor.currentText().startswith('C')
        answers.answers.append(('Average', False, False))
        assert restored.confirm((_detected(tmp_path, 'A', '6.1', 2),))
        assert len(answers.seen) == 5, 'a forgotten file asks again'
        restored.deleteLater()
    finally:
        answers.stop()
        settings.deleteLater()


def test_a_document_without_a_remembered_file_is_unchanged(qapp, tmp_path):
    from xdart.gui.tabs.scattering.point_images_editor import PointImagesSettings
    settings = PointImagesSettings()
    answers = _Answers(qapp, [('Average', False, False)])
    try:
        assert settings.confirm((_detected(tmp_path, 'A', '1.1', 2),)), answers.errors
        assert set(settings.document()) == {'choices'}
    finally:
        answers.stop()
        settings.deleteLater()


def test_an_image_series_detection_names_its_spec_file_and_int_remembers_by_it(qapp, tmp_path):
    from tests.xdart.scattering.test_p1b_output_graph import _write_tiff
    from xdart.gui.tabs.scattering.point_images_editor import PointImagesSettings
    from xrd_tools.sources.point_images import inspect_point_images
    from xrd_tools.sources.selection import image_series_spec
    spec = tmp_path / 'calibrant'
    spec.write_text('#F calibrant\n#E 1\n#O0 del  nu\n\n#S 1 ascan del 10 12 2 1\n#P0 10 0\n#N 3\n'
                    '#L del  nu  i0\n10 0 100\n11 0 200\n'
                    '\n#S 2 ascan del 20 21 1 1\n#P0 20 0\n#N 3\n#L del  nu  i0\n20 0 300\n21 0 400\n')
    for scan in (1, 2):
        for index, value in enumerate((10, 6, 12, 8)):
            _write_tiff(tmp_path / f'calibrant_scan{scan:03}_{index:04}.tif', value)
    first, second = (inspect_point_images(image_series_spec(tmp_path / f'calibrant_scan{scan:03}_0000.tif',
                                                            metadata_format='spec')) for scan in (1, 2))
    assert first is not None and second is not None  # both series name their folder, not their SPEC file
    named = getattr(first, 'metadata_path', None)
    assert named is not None and os.path.samefile(named, spec), named
    settings = PointImagesSettings()
    answers = _Answers(qapp, [('Average', False, True)])
    try:
        assert settings.confirm((first,)), answers.errors
        settings.finish_run()
        assert settings.confirm((second,))
        assert len(answers.seen) == 1, "Int's next scan of the same SPEC file is answered without asking"
    finally:
        answers.stop()
        settings.deleteLater()


def _reject_after(qapp, seconds, seen):
    """A watchdog: whatever question is still open after ``seconds`` is cancelled (the parent never answers)."""
    from xdart.gui.tabs.scattering.point_images_editor import PointImagesDialog
    started = QtCore.QElapsedTimer()
    started.start()
    timer = QtCore.QTimer()

    def tick():
        dialog = qapp.activeModalWidget()
        if isinstance(dialog, PointImagesDialog):
            if dialog not in seen:
                seen.append(dialog)
                seen.append(dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).text())
            if started.elapsed() > seconds * 1000:
                dialog.reject()
    timer.timeout.connect(tick)
    timer.start(10)
    return timer


def test_the_question_confirms_what_it_shows_after_its_countdown(qapp, tmp_path, monkeypatch):
    from xdart.gui.tabs.scattering import point_images_editor
    monkeypatch.setattr(point_images_editor, 'AUTO_CONFIRM_SECONDS', 1, raising=False)
    settings = point_images_editor.PointImagesSettings()
    seen = []
    watchdog = _reject_after(qapp, 6, seen)
    try:
        assert point_images_editor.AUTO_CONFIRM_SECONDS == 1
        assert settings.confirm((_detected(tmp_path, 'A', '1.1', 2), _detected(tmp_path, 'A', '2.1', 3))), seen
        assert len(seen) == 2 and '1 s' in seen[1], seen  # one question, its button counting down
        assert _policies(settings) == {'1.1': ('average', 2), '2.1': ('average', 3)}  # the rest of this run too
        settings.finish_run()
        assert settings.document() == {'choices': settings.document()['choices']}  # nothing remembered by itself
    finally:
        watchdog.stop()
        settings.deleteLater()


def test_a_question_someone_answers_does_not_confirm_itself(qapp, tmp_path, monkeypatch):
    from xdart.gui.tabs.scattering import point_images_editor
    monkeypatch.setattr(point_images_editor, 'AUTO_CONFIRM_SECONDS', 1, raising=False)
    settings = point_images_editor.PointImagesSettings()
    seen = []
    touched = []
    watchdog = _reject_after(qapp, 3, seen)

    def touch():
        dialog = qapp.activeModalWidget()
        if isinstance(dialog, point_images_editor.PointImagesDialog) and not touched:
            touched.append(dialog)
            dialog.operation.editor.setCurrentText('Select one')
            dialog.operation.valueChanged.emit(dialog.operation.path, 'Select one')
    toucher = QtCore.QTimer()
    toucher.timeout.connect(touch)
    toucher.start(5)
    try:
        assert not settings.confirm((_detected(tmp_path, 'A', '1.1', 2),))  # still open at 3 s: cancelled
        assert touched and settings.choices == ()
    finally:
        toucher.stop()
        watchdog.stop()
        settings.deleteLater()


def test_a_scan_answered_again_by_a_remembered_file_asks_instead(qapp, tmp_path):
    """The question was the only brake on resolve → ask → resolve: a scan that comes back with the grouping it was
    already given is asked about, without a countdown, rather than answered again."""
    from xdart.gui.tabs.scattering import point_images_editor
    settings = point_images_editor.PointImagesSettings()
    answers = _Answers(qapp, [('Average', False, True), ('Average', False, False)])
    try:
        assert settings.confirm((_detected(tmp_path, 'A', '1.1', 2),)), answers.errors
        settings.finish_run()
        assert settings.confirm((_detected(tmp_path, 'A', '2.1', 2),))
        assert len(answers.seen) == 1
        settings.finish_run()
        countdown = []
        probe = QtCore.QTimer()
        probe.timeout.connect(lambda: countdown.append(getattr(qapp.activeModalWidget(), '_countdown', None)))
        probe.start(1)
        try:
            assert settings.confirm((_detected(tmp_path, 'A', '2.1', 2),))
        finally:
            probe.stop()
        assert len(answers.seen) == 2, 'the same scan, already given that grouping, is asked about'
        timers = [timer for timer in countdown if timer is not None]
        assert timers and not any(timer.isActive() for timer in timers)
    finally:
        answers.stop()
        settings.deleteLater()
