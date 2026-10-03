"""Point grouping is confirmed on the real Int Run/admission path."""
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets


@pytest.mark.parametrize('accept', [False, True])
def test_int_run_confirms_tiff_spec_grouping_before_output(qapp, tmp_path, accept):
    from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
    from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
    from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xdart.gui.tabs.scattering.point_images_editor import PointImagesDialog
    from xdart.gui.tabs.scattering.state_machine import RunPhase
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.sources.selection import image_series_spec
    from tests.xdart.scattering._e2sd_support import write_poni
    from tests.xdart.scattering.test_e3_join_oracle import _wait
    from tests.xdart.scattering.test_p1b_output_graph import _intent, _write_tiff
    spec = tmp_path / 'calibrant'
    spec.write_text('#F calibrant\n#E 1\n#O0 del  nu\n\n'
                    '#S 1 ascan del 10 12 2 1\n#P0 10 0\n#N 3\n'
                    '#L del  nu  i0\n10 0 100\n11 0 200\n')
    paths = [tmp_path / f'calibrant_scan001_{index:04}.tif' for index in range(4)]
    for path, value in zip(paths, (10, 6, 12, 8)):
        _write_tiff(path, value)
    poni, output = tmp_path / 'cal.poni', tmp_path / 'processed'
    write_poni(poni)
    intent = _intent(paths[0], output, poni)
    intent.source_spec = image_series_spec(paths[0], metadata_format='spec')
    lifecycle = ScatteringCoordinator()
    store = RunIntentStore(intent)
    page = ScatteringWorkspace(intents=store, lifecycle=lifecycle,
                               sources=FilesystemSourceAdapter(), executor=StandardRunExecutor())
    page.show()
    seen, errors = [], []
    timer = QtCore.QTimer()
    def answer():
        dialog = qapp.activeModalWidget()
        if isinstance(dialog, PointImagesDialog) and dialog not in seen:
            seen.append(dialog)
            try:
                assert not list(output.glob('*.nexus'))
                assert (dialog.detection.point_count, dialog.detection.image_count) == (2, 4)
                dialog.operation.editor.setCurrentText('Difference (2 − 1)')
                dialog.operation.valueChanged.emit(dialog.operation.path, 'Difference (2 − 1)')
                button = QtWidgets.QDialogButtonBox.StandardButton.Ok if accept else QtWidgets.QDialogButtonBox.StandardButton.Cancel
                dialog.buttons.button(button).click()
            except Exception as error:
                errors.append(error)
                dialog.reject()
    timer.timeout.connect(answer)
    timer.start(5)
    try:
        _wait(qapp, lambda: page._shell.run_controls.startButton.isEnabled())
        page._shell.run_controls.startButton.click()
        _wait(qapp, lambda: bool(seen), timeout=15)
        assert not errors
        _wait(qapp, lambda: lifecycle.phase in {RunPhase.IDLE, RunPhase.FAILED} and page._admission is None, timeout=20)
        assert len(seen) == 1
        if accept:
            _wait(qapp, lambda: bool(list(output.glob('*.nexus'))), timeout=20,
                  diagnostic=lambda: f'{lifecycle.phase.value}: {page._notice_text}; {page._progress.detail}')
            policy = store.snapshot().thaw().source_spec.options['point_images']
            assert policy == dict(images_per_point=2, operation='difference', image_index=1, reverse=False)
        else:
            assert 'point_images' not in store.snapshot().thaw().source_spec.options
            assert not list(output.glob('*.nexus'))
    finally:
        timer.stop()
        _wait(qapp, lambda: page.close_workspace().cleanup_status.value == 'cleaned')
        page.close()


@pytest.fixture
def qapp(_xdart_qt_harness):
    return _xdart_qt_harness.app
