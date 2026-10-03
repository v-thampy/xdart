"""Confirmed container points keep the real admission, cursor and publication owners."""
import h5py
import numpy as np
import pytest

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io import open_scan
from xrd_tools.session.run_configuration import ThresholdIntent
from xrd_tools.sources.point_images import PointImagePolicy


@pytest.mark.parametrize('operation', ('average', 'difference', 'select'))
def test_confirmed_container_run_publishes_logical_points_and_native_statistics(tmp_path, operation):
    from tests.xdart.scattering._e2sd_support import write_poni
    from tests.xdart.scattering.test_p1b_output_graph import _intent
    from tests.xdart.scattering.test_e1b1_terminal_cleanup import _prepared_run, _terminal
    from xdart.gui.tabs.scattering.display_values import StandardEventKind
    from xdart.gui.tabs.scattering.events import CleanupStatus
    from xrd_tools.sources import open_source
    from xrd_tools.reduction import run_reduction
    from xrd_tools.integrate.calibration import load_detector_calibration, detector_calibration_to_integrator
    from xrd_tools.session.readiness import native_int_reduction_plan

    path, poni = tmp_path / 'points.h5', tmp_path / 'cal.poni'
    images = np.stack([np.full((195, 487), value, np.uint16) for value in (40, 20, 60, 30)])
    images[0, :4] = 65535
    with h5py.File(path, 'w') as stream:
        detector = stream.create_group('entry/instrument/detector')
        detector.create_dataset('data', data=images)
        detector.create_dataset('detectorSpecific/nimages', data=2)
        detector.create_dataset('detectorSpecific/ntrigger', data=2)
    write_poni(poni)
    policy = PointImagePolicy(2, operation, image_index=2)
    spec = SourceSpec(path, SourceKind.NEXUS_STACK, options={'point_images': policy.to_dict()})
    intent = _intent(path, tmp_path / 'processed', poni, processing_mode='Int 2D')
    intent.source_spec = spec
    intent.threshold = ThresholdIntent(apply_threshold=True, threshold_min=5., threshold_max=65534.)
    intent.bai_1d_args['error_model'] = 'poisson'
    intent.bai_2d_args['error_model'] = 'poisson'
    executor, run, _ = _prepared_run(tmp_path, intent=intent)
    configuration = run.configuration
    try:
        executor._run(run)
        terminal = _terminal(executor)
        assert terminal.kind is StandardEventKind.FINISHED, terminal.detail
        saved = open_scan(run.artifact)
        if saved:
            assert saved.frame_indices == [0, 1]
            source = open_source(spec)
            try:
                calibration = load_detector_calibration(poni)
                scan = source.to_scan(poni=calibration.poni, integrator=detector_calibration_to_integrator(calibration))
                plan = native_int_reduction_plan(configuration)
                expected = run_reduction(plan, scan, executor=1)
                for label in (0, 1):
                    view = saved.get_1d(label)
                    np.testing.assert_allclose(view.intensity, expected.frames[label].result_1d.intensity, rtol=1e-6)
                    np.testing.assert_allclose(view.sigma, expected.frames[label].result_1d.sigma, rtol=1e-6)
            finally:
                source.close()
    finally:
        assert executor.close(run.identity).cleanup_status is CleanupStatus.CLEANED
