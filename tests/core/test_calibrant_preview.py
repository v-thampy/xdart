"""A refined calibrant preview is a real Stitch, without publishing or changing inputs."""
from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from tests.core.test_stitch_autocalibration import ROW_ANGLES, five_parameter_seed, ring_scan


@pytest.fixture
def refined_scan(tmp_path):
    from xrd_tools.integrate.stitch_autocalibration import refine_arm_from_scan
    spec = ring_scan(tmp_path / 'scan', [(d, 0.) for d in (6., 20., 34.)])
    fit = refine_arm_from_scan(spec, five_parameter_seed(ROW_ANGLES), max_frames=3)
    return spec, fit


def test_preview_is_the_full_scan_stitched_at_the_fitted_geometry(refined_scan):
    from pyFAI.multi_geometry import MultiGeometry
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.integrate.multi import create_multigeometry_integrator_series_from_geometry
    from xrd_tools.integrate.stitch_calibration import stitch_calibrant_preview
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning, condition_stitch_frame
    from xrd_tools.sources import open_source
    spec, fit = refined_scan
    # A fit can use fewer points/frames; the preview must still stitch the full source.
    fit = replace(fit, asset={**fit.asset, 'provenance': {**fit.asset['provenance'], 'frames': []}})
    band = StitchConditioning('manual', 100., 6000.)
    mask = np.zeros((195, 1475), bool)
    mask[:, :400] = True
    diff = Diffractometer.from_pyfai_goniometer(fit.asset['calibration'],
        source_motors=fit.asset['provenance']['source_motors'])
    source = open_source(spec)
    try:
        motors = {name: np.array([source.metadata_for(i)[name] for i in source.frame_indices])
                  for name in ('del', 'nu')}
        images = [condition_stitch_frame(source.load_native_frame(i), band) for i in source.frame_indices]
    finally:
        close = getattr(source, 'close', None)
        if close is not None:
            close()
    integrators = list(create_multigeometry_integrator_series_from_geometry(diff, motors))
    mg = MultiGeometry(integrators, unit='q_A^-1', radial_range=(.1, 6.),
                       azimuth_range=(-180., 180.), threadpoolsize=0)
    expected = mg.integrate1d(images, 120, method='BBox',
        lst_mask=[mask | np.asarray(ai.detector.calc_mask(), dtype=bool) for ai in integrators])
    original = {path: path.read_bytes() for path in spec.uri.parent.rglob('*') if path.is_file()}
    result = stitch_calibrant_preview(fit, spec, mask=mask, conditioning=band,
        radial_range=(.1, 6.), npt_rad=80, npt_azim=36, npt_1d=120)
    np.testing.assert_array_equal(result.profile[0].radial, expected.radial)
    # pyFAI writes 0 in a bin no pixel reaches; native Stitch profiles mark it NaN (5cc058db), as the preview does
    empty = np.asarray(expected.count) == 0
    assert empty.any()  # the band and mask leave such bins, so the rule is exercised
    np.testing.assert_allclose(result.profile[0].intensity, np.where(empty, np.nan, expected.intensity),
                               rtol=1e-12, atol=1e-12)
    assert result.payload.intensity.shape == (80, 36)
    assert result.provenance['calibrant_preview']['frame_count'] == 3
    assert all(path.read_bytes() == data for path, data in original.items())
    assert set(original) == {path for path in spec.uri.parent.rglob('*') if path.is_file()}


def test_preview_applies_the_camera_once_and_honours_cancellation(refined_scan):
    from xrd_tools.analysis.plans import StitchCancelled
    from xrd_tools.integrate.stitch_calibration import stitch_calibrant_preview
    spec, fit = refined_scan
    plain = stitch_calibrant_preview(fit, spec, npt_rad=80, npt_azim=36, npt_1d=120)
    for path in spec.uri.parent.joinpath('images').glob('*.raw'):
        np.rot90(np.fromfile(path, dtype='<i4').reshape(195, 1475), 2).tofile(path)
    turned = replace(fit, asset={**fit.asset, 'provenance': {**fit.asset['provenance'], 'image_rotation': 180}})
    result = stitch_calibrant_preview(turned, spec, npt_rad=80, npt_azim=36, npt_1d=120)
    np.testing.assert_array_equal(result.payload.intensity, plain.payload.intensity)
    np.testing.assert_array_equal(result.profile[0].intensity, plain.profile[0].intensity)
    cancelled = Event()
    cancelled.set()
    with pytest.raises(StitchCancelled):
        stitch_calibrant_preview(fit, spec, cancelled=cancelled)
