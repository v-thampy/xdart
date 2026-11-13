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
        self.counting = []
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._answer)
        self.timer.start(5)

    def _answer(self):
        from xdart.gui.tabs.scattering.point_images_editor import PointImagesDialog
        dialog = self.qapp.activeModalWidget()
        if not isinstance(dialog, PointImagesDialog) or dialog in self.seen:
            return
        self.seen.append(dialog)
        self.counting.append(dialog._counting)  # as it opened, before this answer changes anything
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


def test_countdown_waits_while_another_modal_covers_the_question(qapp, tmp_path, monkeypatch):
    from xdart.gui.tabs.scattering import point_images_editor
    monkeypatch.setattr(point_images_editor, 'AUTO_CONFIRM_SECONDS', 1)
    dialog = point_images_editor.PointImagesDialog(_detected(tmp_path, 'A', '1.1', 2))
    covered = QtWidgets.QDialog(dialog)
    during_cover = []

    def uncover():
        during_cover.append((dialog.auto_confirmed, dialog.isVisible()))
        covered.reject()

    def cover():
        QtCore.QTimer.singleShot(1300, covered, uncover)
        covered.exec()
        dialog.activateWindow()

    QtCore.QTimer.singleShot(0, dialog, cover)
    QtCore.QTimer.singleShot(5000, dialog, dialog.reject)
    assert dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
    assert during_cover == [(False, True)]
    assert dialog.auto_confirmed


def test_changing_the_operation_stops_the_countdown_before_the_deferred_form_edit(qapp, tmp_path, monkeypatch):
    from PySide6.QtTest import QTest
    from xdart.gui.tabs.scattering import point_images_editor
    monkeypatch.setattr(point_images_editor, 'AUTO_CONFIRM_SECONDS', 1)
    dialog = point_images_editor.PointImagesDialog(_detected(tmp_path, 'A', '1.1', 2))
    counting = []

    def choose():
        dialog.operation.editor.setFocus()
        QTest.keyClick(dialog.operation.editor, QtCore.Qt.Key.Key_Down)
        counting.append(dialog._countdown.isActive())

    QtCore.QTimer.singleShot(0, dialog, choose)
    QtCore.QTimer.singleShot(2200, dialog, dialog.reject)
    assert dialog.exec() == QtWidgets.QDialog.DialogCode.Rejected
    assert dialog.operation.current_value() == 'Difference (2 − 1)'
    assert counting == [False] and not dialog.auto_confirmed


def test_a_disabled_confirmation_cannot_resume_an_elapsed_countdown(qapp, tmp_path, monkeypatch):
    from xdart.gui.tabs.scattering import point_images_editor
    monkeypatch.setattr(point_images_editor, 'AUTO_CONFIRM_SECONDS', 1)
    dialog = point_images_editor.PointImagesDialog(_detected(tmp_path, 'A', '1.1', 2))
    button = dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    button.setEnabled(False)
    QtCore.QTimer.singleShot(1500, dialog, dialog.reject)
    assert dialog.exec() == QtWidgets.QDialog.DialogCode.Rejected
    assert button.text() == 'Confirm grouping'
    button.setEnabled(True)
    QtCore.QTimer.singleShot(1300, dialog, dialog.reject)
    assert dialog.exec() == QtWidgets.QDialog.DialogCode.Rejected
    assert not dialog.auto_confirmed


def test_a_second_grouping_request_cannot_open_over_the_current_question(qapp, tmp_path):
    from xdart.gui.tabs.scattering.point_images_editor import PointImagesDialog, PointImagesSettings
    first, second = PointImagesSettings(), PointImagesSettings()
    answered = _Answers(qapp, [('Average', False, False), ('Average', False, False)])
    result = []

    def request_second():
        if isinstance(qapp.activeModalWidget(), PointImagesDialog) and not result:
            result.append(second.confirm((_detected(tmp_path, 'B', '1.1', 2),)))

    QtCore.QTimer.singleShot(0, first, request_second)
    try:
        assert first.confirm((_detected(tmp_path, 'A', '1.1', 2),))
        assert result == [False] and len(answered.seen) == 1
        assert second.choices == ()
    finally:
        answered.stop()


def test_cancelling_a_later_question_leaves_the_accepted_file_visible(qapp, tmp_path):
    from xdart.gui.tabs.scattering.point_images_editor import PointImagesSettings
    settings = PointImagesSettings()
    answers = _Answers(qapp, [('Average', False, True)])
    try:
        assert not settings.confirm((_detected(tmp_path, 'A', '1.1', 2), _detected(tmp_path, 'B', '1.1', 2)))
        assert settings.document()['file_defaults'][0]['file'] == str(tmp_path / 'A')
        assert settings.remembered.editor.currentText() == 'A · Average'
        assert settings.remember.isEnabled()
    finally:
        answers.stop()


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
        assert settings.confirm((_detected(tmp_path, 'A', '2.1', 2),))
        assert len(answers.seen) == 2, 'the same scan, already given that grouping, is asked about'
        # Recorded as each question opened (hosted CI 43488500230: a 1 ms probe could miss the open question).
        assert answers.counting == [True, False], answers.counting
    finally:
        answers.stop()
        settings.deleteLater()


def test_a_scan_returning_after_its_own_countdown_has_no_second_countdown(qapp, tmp_path, monkeypatch):
    from xdart.gui.tabs.scattering import point_images_editor
    monkeypatch.setattr(point_images_editor, 'AUTO_CONFIRM_SECONDS', 1)
    settings = point_images_editor.PointImagesSettings()
    detection = _detected(tmp_path, 'A', '1.1', 2)
    seen = []
    watchdog = _reject_after(qapp, 4, seen)
    try:
        assert settings.confirm((detection,))
        settings.finish_run()
        assert not settings.confirm((detection,))
        assert seen[-1] == 'Confirm grouping'
    finally:
        watchdog.stop()


def test_ints_profile_keeps_a_remembered_file_and_a_path_written_otherwise_matches(qapp, tmp_path):
    """Int keeps the card's document in the Run's options: a profile carries a remembered file and the card reopens
    it; the same file written another way (a ``..`` here; case and separators on Windows) is the same file."""
    from tests.xdart.scattering.test_p1b_output_graph import _intent
    from xdart.gui.tabs.scattering.point_images_editor import PointImagesSettings
    from xrd_tools.session.run_intent_profile import dump_run_intent_profile, load_run_intent_profile
    settings = PointImagesSettings()
    answers = _Answers(qapp, [('Average', False, True)])
    try:
        assert settings.confirm((_detected(tmp_path, 'Calibrant_SPEC', '1.1', 2),)), answers.errors
        intent = settings.apply_intent(_intent(tmp_path / 'a.tif', tmp_path / 'out', tmp_path / 'cal.poni'))
        loaded = load_run_intent_profile(dump_run_intent_profile(intent))
        restored = PointImagesSettings()
        restored.restore_intent(loaded)
        assert restored.document() == settings.document()
        assert restored.remembered.editor.currentText() == 'Calibrant_SPEC · Average'
        (tmp_path / 'sub').mkdir()
        assert restored.confirm((_detected(tmp_path / 'sub' / '..', 'Calibrant_SPEC', '2.1', 2),))
        assert len(answers.seen) == 1
        restored.deleteLater()
    finally:
        answers.stop()
        settings.deleteLater()


def test_ints_run_drops_a_stored_grouping_its_finished_scan_no_longer_shows(qapp, tmp_path):
    """His pick "Fix first, then push (Recommended)": a grouping stored for a scan answered while it recorded (two
    images per point) no longer fits once the scan reads one image per point; Int's Run forgets it, says so, and
    processes the scan ungrouped, rather than refusing it from then on."""
    from dataclasses import replace
    from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
    from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
    from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xdart.gui.tabs.scattering.state_machine import RunPhase
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.sources.point_images import PointImagePolicy
    from xrd_tools.sources.selection import image_series_spec
    from tests.xdart.scattering._e2sd_support import write_poni
    from tests.xdart.scattering.test_e3_join_oracle import _wait
    from tests.xdart.scattering.test_p1b_output_graph import _intent, _write_tiff
    spec = tmp_path / 'calibrant'
    spec.write_text('#F calibrant\n#E 1\n#O0 del  nu\n\n#S 1 ascan del 10 12 2 1\n#P0 10 0\n#N 3\n'
                    '#L del  nu  i0\n10 0 100\n11 0 200\n')
    paths = [tmp_path / f'calibrant_scan001_{index:04}.tif' for index in range(2)]
    for path, value in zip(paths, (10, 12)):
        _write_tiff(path, value)
    poni, output = tmp_path / 'cal.poni', tmp_path / 'processed'
    write_poni(poni)
    intent = _intent(paths[0], output, poni)
    series = image_series_spec(paths[0], metadata_format='spec')
    intent.source_spec = replace(series, options={**series.options,
                                                  'point_images': PointImagePolicy(2, 'average', 1, False).to_dict()})
    lifecycle = ScatteringCoordinator()
    store = RunIntentStore(intent)
    page = ScatteringWorkspace(intents=store, lifecycle=lifecycle,
                               sources=FilesystemSourceAdapter(), executor=StandardRunExecutor())
    page.show()
    try:
        assert page.point_images.choices  # the stored grouping, as the scan's settings hold it
        notices = []
        page.noticeChanged.connect(notices.append)
        _wait(qapp, lambda: page._shell.run_controls.startButton.isEnabled())
        page._shell.run_controls.startButton.click()
        _wait(qapp, lambda: bool(list(output.glob('*.nexus'))), timeout=30,
              diagnostic=lambda: f'{lifecycle.phase.value}: {page._notice_text}')
        _wait(qapp, lambda: lifecycle.phase in {RunPhase.IDLE, RunPhase.FAILED} and page._admission is None,
              timeout=30)
        assert 'point_images' not in store.snapshot().thaw().source_spec.options
        assert page.point_images.choices == ()
        assert any('no longer fits' in notice for notice in notices), notices
    finally:
        _wait(qapp, lambda: page.close_workspace().cleanup_status.value == 'cleaned')
        page.close()


def _int_page(tmp_path, *, points, images, policy=None, folder=False):
    """Int on a SPEC-backed image series of ``points`` rows and ``images`` files with a stored grouping."""
    from dataclasses import replace
    from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
    from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
    from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.sources.selection import DirectorySourceSpec, image_series_spec
    from tests.xdart.scattering._e2sd_support import write_poni
    from tests.xdart.scattering.test_p1b_output_graph import _intent, _write_tiff
    rows = ''.join(f'{10 + point} 0 {100 * (point + 1)}\n' for point in range(points))
    (tmp_path / 'calibrant').write_text('#F calibrant\n#E 1\n#O0 del  nu\n\n#S 1 ascan del 10 12 2 1\n#P0 10 0\n#N 3\n'
                                        '#L del  nu  i0\n' + rows)
    paths = [tmp_path / f'calibrant_scan001_{index:04}.tif' for index in range(images)]
    for index, path in enumerate(paths):
        _write_tiff(path, 10 + index)
    poni, output = tmp_path / 'cal.poni', tmp_path / 'processed'
    write_poni(poni)
    intent = _intent(paths[0], output, poni)
    series = image_series_spec(paths[0], metadata_format='spec')
    intent.source_spec = series if policy is None else replace(
        series, options={**series.options, 'point_images': policy.to_dict()})
    if folder:
        intent.source_spec = DirectorySourceSpec(tmp_path, suffixes=('.tif',), metadata_format='spec')
        intent.gi.enabled = True
    lifecycle = ScatteringCoordinator()
    store = RunIntentStore(intent)
    page = ScatteringWorkspace(intents=store, lifecycle=lifecycle,
                               sources=FilesystemSourceAdapter(), executor=StandardRunExecutor())
    page.show()
    return page, store, lifecycle, output


def test_int_folder_does_not_keep_retrying_an_unappliable_answer(qapp, tmp_path, monkeypatch):
    from xdart.gui.tabs.scattering import point_images_editor
    from tests.xdart.scattering.test_e3_join_oracle import _wait
    monkeypatch.setattr(point_images_editor, 'AUTO_CONFIRM_SECONDS', 1)
    page, _store, lifecycle, output = _int_page(tmp_path, points=1, images=2, folder=True)
    seen = []
    notices = []
    page.noticeChanged.connect(notices.append)
    watchdog = _reject_after(qapp, 5, seen)
    try:
        _wait(qapp, lambda: page._shell.run_controls.startButton.isEnabled())
        page._shell.run_controls.startButton.click()
        _wait(qapp, lambda: page._admission is None and lifecycle.phase.value in {'idle', 'failed'}
              and any('group' in notice.lower() for notice in notices), timeout=15,
              diagnostic=lambda: f'{lifecycle.phase.value}: {notices}')
        assert len(seen) <= 2, 'a Folder Run must not ask for a grouping it cannot apply on its retry'
        assert any('Folder' in notice for notice in notices), notices
        assert not list(output.glob('*.nexus'))
    finally:
        watchdog.stop()
        _wait(qapp, lambda: page.close_workspace().cleanup_status.value == 'cleaned')
        page.close()


def test_int_image_series_keeps_its_own_confirmed_operation(qapp, tmp_path, monkeypatch):
    from xdart.gui.tabs.scattering import point_images_editor
    from xrd_tools.sources.point_images import inspect_point_images
    from xrd_tools.sources.selection import image_series_spec
    from tests.xdart.scattering.test_p1b_output_graph import _write_tiff
    monkeypatch.setattr(point_images_editor, 'AUTO_CONFIRM_SECONDS', 1)
    spec = tmp_path / 'calibrant'
    spec.write_text('#F calibrant\n#E 1\n#O0 del  nu\n\n#S 1 ascan del 10 11 1 1\n#P0 10 0\n#N 3\n'
                    '#L del  nu  i0\n10 0 100\n11 0 200\n'
                    '\n#S 2 ascan del 20 21 1 1\n#P0 20 0\n#N 3\n#L del  nu  i0\n20 0 300\n21 0 400\n')
    for scan in (1, 2):
        for index in range(4):
            _write_tiff(tmp_path / f'calibrant_scan{scan:03}_{index:04}.tif', 10 + index)
    first, second = (inspect_point_images(image_series_spec(
        tmp_path / f'calibrant_scan{scan:03}_0000.tif', metadata_format='spec')) for scan in (1, 2))
    settings = point_images_editor.PointImagesSettings()
    answers = _Answers(qapp, [('Difference (2 − 1)', False, False), ('Average', False, False)])
    try:
        assert settings.confirm((first,))
        assert settings.confirm((second,))
    finally:
        answers.stop()
    settings.restore(settings.document())
    seen = []
    watchdog = _reject_after(qapp, 5, seen)
    try:
        assert settings.confirm((first,))
        assert settings.options(first.source_spec).options['point_images']['operation'] == 'difference'
        assert settings.options(second.source_spec).options['point_images']['operation'] == 'average'
    finally:
        watchdog.stop()


def test_int_does_not_drop_a_grouping_from_a_frozen_partial_image_series(qapp, tmp_path):
    from xrd_tools.sources.point_images import PointImagePolicy
    from tests.xdart.scattering.test_e3_join_oracle import _wait
    from tests.xdart.scattering.test_p1b_output_graph import _write_tiff
    policy = PointImagePolicy(2, 'difference')
    page, store, lifecycle, output = _int_page(tmp_path, points=2, images=2, policy=policy)
    notices = []
    page.noticeChanged.connect(notices.append)
    seen = []
    watchdog = _reject_after(qapp, 5, seen)
    try:
        for index in (2, 3):
            _write_tiff(tmp_path / f'calibrant_scan001_{index:04}.tif', 10 + index)
        _wait(qapp, lambda: page._shell.run_controls.startButton.isEnabled())
        page._shell.run_controls.startButton.click()
        _wait(qapp, lambda: page._admission is None and lifecycle.phase.value in {'idle', 'failed'}
              and bool(notices), timeout=20, diagnostic=lambda: f'{lifecycle.phase.value}: {notices}')
        assert store.snapshot().thaw().source_spec.options.get('point_images') == policy.to_dict()
        assert not list(output.glob('*.nexus')) and not seen
    finally:
        watchdog.stop()
        _wait(qapp, lambda: page.close_workspace().cleanup_status.value == 'cleaned')
        page.close()


@pytest.mark.parametrize('case', ['counts in flux', 'SPEC file not found'])
def test_ints_run_keeps_a_stored_grouping_it_cannot_show_is_stale(qapp, tmp_path, case):
    """The rule forgets a stored grouping only on positive evidence that the scan reads one image per point: counts
    that are not yet a whole multiple (three points, four images), or a scan whose SPEC file cannot be found, refuse
    the Run as before, keep the grouping (a Difference here, which a later question would offer as Average) and ask
    nothing."""
    from xdart.gui.tabs.scattering.point_images_editor import PointImagesDialog
    from xdart.gui.tabs.scattering.state_machine import RunPhase
    from xrd_tools.sources.point_images import PointImagePolicy
    from tests.xdart.scattering.test_e3_join_oracle import _wait
    policy = PointImagePolicy(2, 'difference', 1, False)
    page, store, lifecycle, output = _int_page(tmp_path, points=3, images=4, policy=policy)
    asked = []
    probe = QtCore.QTimer()

    def watch():
        dialog = qapp.activeModalWidget()
        if isinstance(dialog, PointImagesDialog):
            asked.append(dialog)
            dialog.reject()
    probe.timeout.connect(watch)
    probe.start(5)
    try:
        if case == 'SPEC file not found':
            (tmp_path / 'calibrant').rename(tmp_path / 'calibrant.moved')
        notices = []
        page.noticeChanged.connect(notices.append)
        _wait(qapp, lambda: page._shell.run_controls.startButton.isEnabled())
        page._shell.run_controls.startButton.click()
        _wait(qapp, lambda: (any('admission failed' in notice.lower() for notice in notices)
                             or list(output.glob('*.nexus')))
              and lifecycle.phase in {RunPhase.IDLE, RunPhase.FAILED} and page._admission is None, timeout=30,
              diagnostic=lambda: f'{lifecycle.phase.value}: {notices}')
        assert not list(output.glob('*.nexus')) and not asked, notices
        assert not any('no longer fits' in notice for notice in notices), notices
        assert store.snapshot().thaw().source_spec.options['point_images'] == policy.to_dict()
        assert [stored.operation for _uri, _scan, stored in page.point_images.choices] == ['difference']
    finally:
        probe.stop()
        _wait(qapp, lambda: page.close_workspace().cleanup_status.value == 'cleaned')
        page.close()
