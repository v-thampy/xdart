"""Real artifact -> shared RSM fit windows -> numerical worker -> shared display."""
import threading
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from tests.core.test_analysis_artifact import _digest, _rsm_execution_attestation
from xrd_tools.analysis.fitting.region import PeakSeed, RegionAxis
from xrd_tools.analysis.fitting.region_analyzers import REGION_3D_KEY, Region3DFitAnalyzer, Region3DInput
from xrd_tools.analysis.runner import AnalysisInput
from xrd_tools.io.analysis_artifact import (
    ANALYSIS_SCHEMA_VERSION_V2, AnalysisArtifactKind, AnalysisArtifactOverwrite, AnalysisArtifactRequest,
    admit_analysis_artifact, analysis_execution_attestation_digest, project_analysis_artifact_result,
    read_analysis_artifact,
)
from xrd_tools.io.nexus import write_rsm
from xrd_tools.io.output_transaction import OutputTransactionCoordinator


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def payload(tmp_path, shape=(13, 15, 17)):
    axes = tuple((name, np.linspace(-2, 2, count)) for name, count in zip(('h', 'k', 'l'), shape))
    xyz = np.meshgrid(*(axis for _, axis in axes), indexing='ij')
    data = 3 + .1*xyz[0] - .2*xyz[1] + .05*xyz[2]
    data += 80*np.exp(-sum((coordinate-center)**2/width**2 for coordinate, center, width
                           in zip(xyz, (.15, -.2, .1), (.45, .65, .55))))
    projection = project_analysis_artifact_result(kind=AnalysisArtifactKind.RSM, axes=axes,
        axis_units=(('h', None), ('k', None), ('l', None)), intensity=data,
        sigma=None, coverage=None, normalization=None)
    request_id = _digest('fit-window-request')
    attestation = _rsm_execution_attestation(request_id, projection.result_fingerprint)
    request = AnalysisArtifactRequest(tmp_path/'region.nexus', AnalysisArtifactKind.RSM,
        AnalysisArtifactOverwrite.CREATE_NEW, request_id, _digest('source'), _digest('plan'),
        _digest('provenance'), {'schema_version': 'rsm-operation-v2-intent'},
        schema_version=ANALYSIS_SCHEMA_VERSION_V2, execution_attestation=attestation,
        execution_attestation_digest=analysis_execution_attestation_digest(
            AnalysisArtifactKind.RSM, attestation, request_fingerprint=request_id))
    receipt = admit_analysis_artifact(request, coordinator=OutputTransactionCoordinator()).publish(
        lambda entry: write_rsm(entry, result_projection=projection,
                                provenance=request.provenance_json, bounded_artifact=True))
    return read_analysis_artifact(request.target, expected_receipt=receipt)


def settle(qapp, window):
    limit = time.monotonic() + 30
    while window.active() and time.monotonic() < limit:
        qapp.processEvents()
        time.sleep(.01)
    assert not window.active()


def close(qapp, window):
    from xdart.gui.pages.values import PageCleanup
    window.release()
    settle(qapp, window)
    assert window.shutdown().status is PageCleanup.CLEAN
    window.deleteLater()
    qapp.processEvents()


def test_region_analyzer_forwards_real_solver_cancellation():
    axes = tuple(RegionAxis(name, np.linspace(-2, 2, 11), 'r.l.u.') for name in ('h', 'k', 'l'))
    xyz = np.meshgrid(*(axis.values for axis in axes), indexing='ij')
    data = 3 + 50*np.exp(-sum(coordinate**2 for coordinate in xyz))
    cancel = threading.Event()
    cancel.set()
    outcome = Region3DFitAnalyzer(seeds=(PeakSeed('p1', (0., 0., 0.)),), cancel_token=cancel).analyze(
        AnalysisInput('cancel', np.empty(0), np.empty(0), metadata={REGION_3D_KEY: Region3DInput(data, axes)}))
    assert not outcome.ok and 'cancelled' in outcome.message


def test_joint_3d_fit_navigation_preserves_outcome_and_matched_signed_comparison(qapp, tmp_path):
    from xdart.gui.analysis.region_fit_window import RegionFitWindow
    from xdart.gui.tabs.scattering.scientific_view import ScientificView
    from xdart.gui.tabs.scattering.results_table import ResultsTable
    from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
    window = RegionFitWindow(ndim=3)
    try:
        admitted = payload(tmp_path)
        identity = str(tmp_path / 'region.nexus')
        window.show_result(identity, admitted)
        settle(qapp, window)
        assert window.outcome is None
        assert window.current().load().key.path == identity
        assert window.current().reference.path == identity
        for suffix in ('ipynb', 'py'):
            destination = tmp_path / f'before_fit.{suffix}'
            window.export_to(destination)
            settle(qapp, window)
            assert destination.is_file(), window.card.status.text()
            assert 'rsm_region' in destination.read_text('utf-8')
        assert not (tmp_path/'before_fit.input.npz').exists()
        window.fit()
        settle(qapp, window)
        assert isinstance(window.view, ScientificView) and isinstance(window.table, ResultsTable)
        outcome = window.outcome
        assert outcome is not None and outcome.ndim == 3 and window.table.rowCount() > 0, window.card.status.text()
        np.testing.assert_allclose([item.value for item in outcome.components[0].center], (.15, -.2, .1), atol=.002)
        assert outcome.physical_reason and window.table.columnCount() == 4
        window.export_csv(tmp_path/'parameters.csv')
        csv = (tmp_path/'parameters.csv').read_text('utf-8')
        assert 'stderr' in csv and 'region.nexus' in csv and 'cov_h_k' in csv
        window.show_result(identity, window._payload)  # repeated page presentation notification
        assert window.outcome is outcome
        window.view.commandRequested.emit(ShellCommand(ShellCommandKind.SET_PANEL_INDEX, value=3, path=('data',)))
        settle(qapp, window)
        assert window.outcome is outcome
        projection = window.view.keyed_projection
        data, model, residual = projection.panels
        assert data.index == model.index == residual.index == 3
        np.testing.assert_allclose(data.image-model.image, residual.image, atol=1e-12)
        window.view.commandRequested.emit(ShellCommand(ShellCommandKind.SET_LOG_SCALE, value=True))
        settle(qapp, window)
        assert window.view.keyed_projection.panels[2].presentation == 'signed'
        assert window.outcome is outcome
    finally:
        close(qapp, window)


def test_2d_rectangle_edit_invalidates_fit_and_uses_native_coordinates(qapp, tmp_path):
    from xdart.gui.analysis.region_fit_window import RegionFitWindow
    window = RegionFitWindow(ndim=2)
    try:
        window.show_result('plane', payload(tmp_path))
        settle(qapp, window)
        window.fit()
        settle(qapp, window)
        assert window.outcome is not None and window.outcome.ndim == 2, window.card.status.text()
        key = window.view.keyed_projection.panels[0].key
        roi = window.view.keyed_panel(key).region
        roi.setPos((-.9, -.8))
        settle(qapp, window)
        assert window.outcome is None and window.table.rowCount() == 0
        assert dict(window.selection())['bounds'] != (('h', (-2., 2.)), ('k', (-2., 2.)), ('l', (-2., 2.)))
    finally:
        close(qapp, window)


def test_oversized_box_refuses_and_release_drops_running_fit(qapp, tmp_path):
    from xdart.gui.analysis.region_fit_window import RegionFitWindow
    window = RegionFitWindow(ndim=3)
    try:
        window.show_result('large', payload(tmp_path, (47, 49, 51)))
        assert not window.active() and 'POINT_LIMIT' in window.card.status.text()
        for name in ('h', 'k', 'l'):
            low, high = window.range_rows[name].editors
            low.setText('-.8')
            high.setText('.8')
        window.fit()
        window.release()
        settle(qapp, window)
        assert window.outcome is None and window.view.keyed_projection is None
    finally:
        close(qapp, window)


def test_lineout_mounts_peak_dialog_and_same_native_snapshot_for_fit(qapp, tmp_path):
    from xdart.gui.tabs.rsm.fit_windows import RSMLineoutWindow
    from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
    from xrd_tools.analysis.fit_sources import rsm_lineout_snapshot
    window = RSMLineoutWindow()
    try:
        admitted = payload(tmp_path)
        root = tmp_path / 'project'
        window.configure_batch(lambda selection: (window.current(),), root)
        identity = str(tmp_path / 'region.nexus')
        window.show_result(identity, admitted)
        settle(qapp, window)
        assert isinstance(window.fit_dialog, PeakFitDialog)
        target = window.current()
        actual = target.load()
        assert actual.key.path == actual.key.result_identity == identity
        assert target.reference.path == identity
        expected = rsm_lineout_snapshot(admitted, actual.key, axis='h', centre=(0., 0., 0.), width=(0., 0., 0.))
        np.testing.assert_array_equal(actual.values, expected.values)
        window.fit_dialog.auto_check.setChecked(False)
        window.fit_dialog.model_combo.setCurrentIndex(0)
        window.fit_dialog.fit_btn.click()
        settle(qapp, window)
        assert window.host.outcomes and window.host.outcomes[0].ok, window.fit_dialog.status.text()
        window.configure_batch(lambda selection: (window.current(),), tmp_path / 'other_project')
        settle(qapp, window)
        assert window.host.source_root == tmp_path / 'other_project'
        assert not window.host.outcomes and window.host.snapshot is not None
        assert window.fit_dialog.batch_btn.isEnabled()
        before = window.scope()
        window.rows['width_k'].editor.setText('.8')
        window.rows['width_k'].editor.editingFinished.emit()
        settle(qapp, window)
        assert window.scope() != before
    finally:
        close(qapp, window)
