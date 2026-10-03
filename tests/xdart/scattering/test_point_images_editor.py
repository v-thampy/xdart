"""The shared Int owner for confirmed point grouping, also mounted by Stitch/RSM."""
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets


@pytest.fixture
def qapp(_xdart_qt_harness):
    return _xdart_qt_harness.app


def test_confirmation_is_explicit_and_difference_selection_are_bounded(qapp, tmp_path):
    from xdart.gui.tabs.scattering.point_images_editor import PointImagesDialog
    from xdart.gui.tabs.scattering.advanced_editor import AdvancedDialog
    from xrd_tools.sources.point_images import PointImageDetection
    from xrd_tools.core.scan import SourceKind, SourceSpec
    spec = SourceSpec(tmp_path / 'scan', SourceKind.SPEC, options={'scan': '1.1'})
    detection = PointImageDetection('scan 1.1', 3, 6, 2, spec)
    dialog = PointImagesDialog(detection)
    try:
        assert isinstance(dialog, AdvancedDialog)
        from xdart.gui.widgets.controls_panel import FormRow
        assert isinstance(dialog.apply_remaining, FormRow)
        assert not dialog.apply_remaining.current_value()
        assert dialog.policy().operation == 'average'
        dialog.operation.editor.setCurrentText('Difference (2 − 1)')
        dialog.operation.valueChanged.emit(dialog.operation.path, 'Difference (2 − 1)')
        assert dialog.policy().operation == 'difference'
        dialog.reverse.editor.click()
        assert dialog.policy().reverse
        dialog.reject()
        assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected
        for control in (dialog.operation, dialog.index, dialog.reverse, dialog.apply_remaining):
            assert control.toolTip()
    finally:
        dialog.deleteLater()


def test_remaining_choice_is_compatible_only_and_does_not_survive_the_run(qapp, tmp_path):
    from xdart.gui.tabs.scattering.point_images_editor import PointImagesDialog, PointImagesSettings
    from xrd_tools.sources.point_images import PointImageDetection
    from xrd_tools.core.scan import SourceKind, SourceSpec
    settings = PointImagesSettings()
    answers = [('Difference (2 − 1)', True), ('Select one', True), ('Average', False)]
    seen = []
    errors = []
    timer = QtCore.QTimer()
    def answer():
        dialog = qapp.activeModalWidget()
        if isinstance(dialog, PointImagesDialog) and dialog not in seen:
            seen.append(dialog)
            try:
                operation, remaining = answers[len(seen) - 1]
                dialog.operation.editor.setCurrentText(operation)
                dialog.operation.valueChanged.emit(dialog.operation.path, operation)
                control = getattr(dialog.apply_remaining, 'editor', dialog.apply_remaining)
                control.setChecked(remaining)
                dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()
            except Exception as error:
                errors.append(error)
                dialog.reject()
    timer.timeout.connect(answer)
    timer.start(5)
    def detected(scan, count):
        spec = SourceSpec(tmp_path / 'scan', SourceKind.SPEC, options={'scan': scan})
        return PointImageDetection(scan, 2, 2 * count, count, spec)
    try:
        assert settings.confirm((detected('1.1', 2), detected('2.1', 2), detected('3.1', 3)))
        assert not errors
        assert len(seen) == 2
        assert [policy.operation for _, _, policy in settings.choices] == ['difference', 'difference', 'select']
        document = settings.document()
        settings.finish_run()
        assert settings.confirm((detected('4.1', 3),))
        assert len(seen) == 3
        restored = PointImagesSettings()
        restored.restore(document)
        assert restored.document() == document
        restored.scan.editor.setCurrentIndex(2)
        restored.scan.valueChanged.emit(restored.scan.path, '3.1')
        assert restored.operation.editor.findText('Difference (2 − 1)') == -1
        assert restored.index.editor.text() == '1'
        restored.deleteLater()
    finally:
        timer.stop()
        settings.deleteLater()


def test_invalid_selected_index_cannot_confirm(qapp, tmp_path):
    from xdart.gui.tabs.scattering.point_images_editor import PointImagesDialog
    from xrd_tools.sources.point_images import PointImageDetection
    from xrd_tools.core.scan import SourceKind, SourceSpec
    spec = SourceSpec(tmp_path / 'scan', SourceKind.SPEC, options={'scan': '1.1'})
    dialog = PointImagesDialog(PointImageDetection('1.1', 2, 6, 3, spec))
    try:
        dialog.operation.editor.setCurrentText('Select one')
        dialog.operation.valueChanged.emit(dialog.operation.path, 'Select one')
        dialog.index.editor.setText('4')
        assert dialog.index.refusal()
        assert not dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).isEnabled()
        dialog.index.editor.setText('3')
        assert dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).isEnabled()
    finally:
        dialog.deleteLater()


def test_three_images_have_average_select_and_no_difference(qapp, tmp_path):
    from xdart.gui.tabs.scattering.point_images_editor import PointImagesDialog
    from xrd_tools.sources.point_images import PointImageDetection
    from xrd_tools.core.scan import SourceKind, SourceSpec
    spec = SourceSpec(tmp_path / 'scan', SourceKind.SPEC, options={'scan': '1.1'})
    dialog = PointImagesDialog(PointImageDetection('scan 1.1', 3, 9, 3, spec))
    try:
        assert dialog.operation.editor.findText('Difference (2 − 1)') == -1
        dialog.operation.editor.setCurrentText('Select one')
        dialog.operation.valueChanged.emit(dialog.operation.path, 'Select one')
        dialog.index.editor.setText('3')
        assert dialog.policy().image_index == 3
        assert not dialog.index.refusal()
    finally:
        dialog.deleteLater()
