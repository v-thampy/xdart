"""Frozen P3b coordinate oracle: real publication and independent pyFAI geometry."""
import inspect
import math
from dataclasses import replace

import numpy as np
import pytest

from tests.core.test_predicted_ring_overlays import calibration
from xrd_tools.reduction import (Frame, Scan, ReductionPlan, Integration1DPlan,
                                 Integration2DPlan, MemorySink)
from xrd_tools.session import FrameRecordStore
from xrd_tools.session.headless_scan import open_headless_scan_session
from xrd_tools.session.readiness import bind_native_int_source_energy, effective_ring_geometry


def published(*, unit='q_A^-1', offsets=(17., 43.), energies=(16., 17.)):
    frames = [Frame(i, image=np.ones((195, 487), dtype=np.float32)*100,
                    metadata={'energy': e}) for i, e in enumerate(energies)]
    scan = Scan('grid', frames)
    accepted = bind_native_int_source_energy(scan, calibration())
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(npt=64, unit=unit, azimuth_offset=offsets[0]),
        integration_2d=Integration2DPlan(npt_rad=64, npt_azim=360, unit=unit,
                                       azimuth_offset=offsets[1]))
    args = {'plan': plan} if 'plan' in inspect.signature(effective_ring_geometry).parameters else {}
    snapshot = effective_ring_geometry(accepted, integrator=scan.integrator, **args)
    store = FrameRecordStore()
    session = open_headless_scan_session(scan, plan, sink=MemorySink(), record_store=store,
                                         executor=1, ring_geometry=snapshot)
    for frame in frames:
        session.submit(frame)
    session.finish()
    views = [next(iter(store.get(f.index).results_1d.values())) for f in frames]
    return scan, views, store


def convention():
    scan, views, store = published()
    snap = views[0].geometry.poni
    # Every K row starts with existing publication behaviour, not a missing API.
    assert snap.get('azimuth_offset_1d') == 17., 'Run omitted its effective 1D azimuth offset'
    assert snap.get('azimuth_offset_2d') == 43.
    assert snap.get('domain') == 'Standard' and snap.get('chiDiscAtPi') is True
    return scan, views, store


def test_run_publishes_the_effective_standard_cake_convention():
    scan, views, store = convention()
    with pytest.raises(TypeError):
        views[0].geometry.poni['azimuth_offset_1d'] = 0
    assert 'ring_geometry' not in scan.extra
    for record in store.snapshot().values():
        for view in (*record.results_1d.values(), *record.results_2d.values()):
            assert view.geometry.poni['azimuth_offset_2d'] == 43.
            assert all(not isinstance(v, (dict, list, np.ndarray)) for v in view.geometry.poni.values())


@pytest.mark.parametrize('orientation', (1, 2, 3, 4))
@pytest.mark.parametrize('unit', ('q_A^-1', '2th_deg'))
def test_q_contours_match_independent_pyfai_at_each_bound_energy(orientation, unit, monkeypatch):
    convention()
    from xrd_tools.geometry_figures import predict_detector_grid
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator
    from xrd_tools.core.geometry.diffractometer import DetectorCalibration
    from contourpy import contour_generator
    from scipy.spatial import cKDTree
    scan, views, _ = published(unit=unit)
    signatures = []
    for view in views:
        cal = calibration(rot1=.07, rot2=-.04, poni1=.006, poni2=.06,
                          wavelength=view.geometry.poni['wavelength'])
        cal = DetectorCalibration(cal.poni, {'orientation': orientation})
        ai = detector_calibration_to_integrator(cal)
        snapshot = dict(effective_ring_geometry(cal, integrator=ai),
                        azimuth_offset_1d=17., azimuth_offset_2d=43., domain='Standard', chiDiscAtPi=True)
        angles = ai.center_array((195, 487), unit='2th_rad', scale=False)
        contours = contour_generator(z=angles)
        def forbidden(*a, **k):
            raise AssertionError('grid prediction allocated a detector-sized angle map')
        with monkeypatch.context() as patch:
            patch.setattr(type(ai), 'center_array', forbidden)
            patch.setattr(type(ai), 'array_from_unit', forbidden)
            grid = predict_detector_grid(snapshot, shape=(195, 487), choice='q', step_px=.4)
        assert grid.curves
        signatures.append(grid.curves[0].xy)
        for curve in grid.curves:
            theta = math.radians(curve.two_theta_deg)
            lines = contours.lines(theta)
            cloud = np.concatenate([np.concatenate([
                np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a)/.1))+1))
                for a, b in zip(line[:-1], line[1:])]) for line in lines])
            assert cKDTree(cloud).query(curve.xy)[0].max() <= .5
            residual = np.abs(ai.tth(curve.xy[:, 1], curve.xy[:, 0])-theta)
            assert residual.max()/(ai.detector.pixel1/ai.dist) <= .05
            assert not curve.xy.flags.writeable
        with pytest.raises(ValueError):
            predict_detector_grid(snapshot, shape=(195, 487), q_levels=(-1.,))
    assert not np.array_equal(*signatures)


@pytest.mark.parametrize('orientation', (1, 2, 3, 4))
def test_chi_half_rays_match_pyfai_without_the_antipode_or_beam(orientation):
    convention()
    from xrd_tools.geometry_figures import predict_detector_grid, detector_grid_unavailable
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator
    from xrd_tools.core.geometry.diffractometer import DetectorCalibration
    cal = calibration(rot1=.15, rot2=-.08, poni1=.01, poni2=.035)
    cal = DetectorCalibration(cal.poni, {'orientation': orientation})
    ai = detector_calibration_to_integrator(cal)
    snap = dict(effective_ring_geometry(cal, integrator=ai), domain='Standard', chiDiscAtPi=True,
                azimuth_offset_1d=17., azimuth_offset_2d=43.)
    for mode, offset in (('1d', 17.), ('2d', 43.)):
        grid = predict_detector_grid(snap, shape=(195, 487), choice='χ', mode=mode, step_px=.4)
        assert grid.curves
        for curve in grid.curves:
            label = float(curve.name.split()[1])
            raw = ai.chi(curve.xy[:, 1], curve.xy[:, 0])
            difference = np.angle(np.exp(1j*(raw-np.deg2rad(label-offset))))
            assert np.abs(difference).max() < 1e-6
            assert np.all(ai.tth(curve.xy[:, 1], curve.xy[:, 0]) > 1e-10)
            assert np.linalg.norm(np.diff(curve.xy, axis=0), axis=1).max() <= .5
    assert detector_grid_unavailable(dict(snap, chiDiscAtPi=False), (195, 487))
    assert detector_grid_unavailable(dict(snap, domain='Grazing'), (195, 487))


def test_nonzero_offset_grid_agrees_with_the_real_standard_cake():
    _, _, store = convention()
    from xrd_tools.geometry_figures import predict_detector_grid
    view = next(iter(store.get(0).results_2d.values()))
    grid = predict_detector_grid(view.geometry.poni, shape=(195, 487), choice='χ')
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator
    ai = detector_calibration_to_integrator(calibration(wavelength=view.geometry.poni['wavelength']))
    image = np.zeros((195, 487))
    for curve in grid.curves:
        level = float(curve.name.split()[1])
        # The physical ray must land in the corresponding real cake bin after its offset.
        x, y = np.rint(curve.xy[len(curve.xy)//2]).astype(int)
        image.fill(0); image[y, x] = 1000
        cake = ai.integrate2d(image, 64, 360, unit='q_A^-1', method='histogram',
                             correctSolidAngle=False)
        j = np.argmax(cake.intensity.sum(axis=1))
        centres = cake.azimuthal+43.
        width = abs(centres[1]-centres[0])
        assert abs(centres[j]-level) <= width*1.5
        zero = predict_detector_grid(dict(view.geometry.poni, azimuth_offset_2d=0.),
                                     shape=(195, 487), choice='χ')
        assert all(float(c.name.split()[1]) % 30 == 0 for c in zero.curves)


def test_default_levels_and_labels_are_finite_physical_and_bounded():
    _, views, _ = convention()
    from xrd_tools.geometry_figures import predict_detector_grid
    snap = views[0].geometry.poni
    grid = predict_detector_grid(snap, shape=(195, 487))
    qnames = {c.name.split(',')[0] for c in grid.curves if c.name.startswith('q ')}
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator
    ai = detector_calibration_to_integrator(calibration(wavelength=snap['wavelength']))
    yy, xx = np.indices((195, 487))
    physical_q = 4*np.pi*np.sin(ai.tth(yy, xx)/2)/(snap['wavelength']*1e10)
    lo, hi = float(physical_q.min()), float(physical_q.max())
    permitted = (multiple*10.**exponent for exponent in range(-12, 13)
                 for multiple in (1, 2, 2.5, 5))
    counts = [(max(0, math.floor(hi/step)-max(1, math.ceil(lo/step))+1), step)
              for step in permitted]
    expected_count, _ = min(counts, key=lambda item: (abs(item[0]-10), -item[1]))
    assert len(qnames) == expected_count
    assert len(grid.curves) <= 60 and len(grid.texts) <= len(grid.curves)
    assert grid.image is None
    assert len({text.name for text in grid.texts}) == len(grid.texts)
    for text in grid.texts:
        x, y = text.xy
        assert min(abs(x), abs(x-486), abs(y), abs(y-194)) < 1e-6
        assert text.tooltip
    for curve in grid.curves:
        assert np.isfinite(curve.xy).all() and len(curve.xy) < 10000
        assert (curve.xy >= -1e-7).all()
        assert (curve.xy <= (486.0000001, 194.0000001)).all()
