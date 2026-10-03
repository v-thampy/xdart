"""Confirmed container points keep the real admission, cursor and publication owners."""
from dataclasses import replace

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
    intent = _intent(path, tmp_path / 'processed', poni, processing_mode='Int 1D + 2D')
    intent.source_spec = spec
    intent.threshold = ThresholdIntent(apply_threshold=True, threshold_min=5., threshold_max=65534.)
    intent.bai_1d_args['error_model'] = 'poisson'
    intent.bai_2d_args['error_model'] = 'poisson'
    executor, run, _ = _prepared_run(tmp_path, intent=intent)
    try:
        executor._run(run)
        terminal = _terminal(executor)
        assert terminal.kind is StandardEventKind.FINISHED, terminal.detail
        with open_scan(run.artifact) as saved:
            assert saved.frame_indices == [0, 1]
            source = open_source(spec)
            try:
                scan = source.to_scan(poni=saved.poni, integrator=saved.integrator)
                plan = run.plan
                expected = run_reduction(plan, scan, executor=1)
                for label in (0, 1):
                    view = saved.get_frame(label)
                    np.testing.assert_allclose(view.int_1d.norm, expected.frames[label].int_1d.norm, rtol=1e-6)
                    assert view.metadata['point_images']['operation'] == operation
            finally:
                source.close()
    finally:
        assert executor.close(run.identity).cleanup_status is CleanupStatus.CLEANED
