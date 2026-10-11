"""P3a frozen oracle: independent pyFAI contours and real binding/publication.

No mockup is an oracle. Missing new APIs earn no behavioral parent-failure credit.
"""
from dataclasses import replace
import inspect
import json
from types import MappingProxyType

import numpy as np
import pytest

from xrd_tools.core.containers import PONI
from xrd_tools.core.geometry.diffractometer import DetectorCalibration
from xrd_tools.integrate.calibration import detector_calibration_to_integrator
from xrd_tools.reduction import (Frame, Scan, ReductionPlan, Integration1DPlan,
                                 Integration2DPlan, GIMode, MemorySink)
from xrd_tools.session import FrameRecordStore
from xrd_tools.session.headless_scan import open_headless_scan_session
from xrd_tools.session.readiness import bind_native_int_source_energy


def calibration(**values):
    base = dict(dist=.03, poni1=.01677, poni2=.041882, wavelength=1e-10,
                detector='Pilatus100k')
    base.update(values)
    return DetectorCalibration(PONI(**base), {'orientation': 3})


def bound_session(energies=(16., 16.), unit='q_A^-1', gi=None, publish=True, tripwire=None):
    import xrd_tools.session.readiness as readiness
    frames = [Frame(label, image=np.full((195, 487), 100., dtype=np.float32),
                    metadata={'energy': energy})
              for label, energy in zip((7, 11, 15, 19), energies)]
    scan = Scan('rings', frames)
    accepted = bind_native_int_source_energy(scan, calibration(wavelength=12.398419843320026/15*1e-10))
    plan = ReductionPlan(integration_1d=Integration1DPlan(npt=64, unit=unit),
                         integration_2d=Integration2DPlan(npt_rad=64, npt_azim=36, unit=unit), gi=gi)
    store = FrameRecordStore()
    kwargs = {}
    # On the parent this drives the exact old session, and fails on published behavior.
    if publish and 'ring_geometry' in inspect.signature(open_headless_scan_session).parameters:
        kwargs['ring_geometry'] = readiness.effective_ring_geometry(accepted, integrator=scan.integrator)
    session = open_headless_scan_session(scan, plan, sink=MemorySink(), record_store=store,
                                          executor=1, **kwargs)
    if tripwire is not None:
        actual = session._upsert_record_store
        def guarded(*args, **kw):
            with tripwire.context() as patch:
                def forbidden(*a, **k):
                    raise AssertionError('publication constructed or serialized scientific geometry')
                patch.setattr('xrd_tools.integrate.calibration.detector_calibration_to_integrator', forbidden)
                patch.setattr('pyFAI.integrator.azimuthal.AzimuthalIntegrator.__init__', forbidden)
                patch.setattr(json, 'dumps', forbidden)
                return actual(*args, **kw)
        session._upsert_record_store = guarded
    for frame in frames:
        session.submit(frame)
    session.finish()
    views = [next(iter(store.get(frame.index).results_1d.values())) for frame in frames]
    return accepted, scan, views, store


def test_live_frameview_carries_the_effective_bound_calibration(monkeypatch):
    accepted, scan, views, store = bound_session(tripwire=monkeypatch)
    assert views[0].geometry is not None and views[0].geometry.poni is not None
    snapshot = views[0].geometry.poni
    assert snapshot['wavelength'] == accepted.poni.wavelength
    assert snapshot['rot1'] == accepted.poni.rot1
    assert json.loads(snapshot['detector_config_json']) == accepted.detector_config
    assert snapshot == views[1].geometry.poni
    assert snapshot['detector_config_json'] is views[1].geometry.poni['detector_config_json']
    for record in store.snapshot().values():
        for view in (*record.results_1d.values(), *record.results_2d.values()):
            assert view.geometry.poni == snapshot
    with pytest.raises(TypeError):
        snapshot['wavelength'] = 1
    assert all(not isinstance(v, (dict, list, np.ndarray)) for v in snapshot.values())
    assert 'ring_geometry' not in scan.extra
    _, _, absent, _ = bound_session(publish=False)
    assert all(v.geometry is None for v in absent)


@pytest.mark.parametrize('unit', ('q_A^-1', '2th_deg'))
def test_energy_scan_geometry_uses_each_frames_bound_wavelength(unit):
    from xrd_tools.core.energy import energy_eV_to_wavelength_m
    _, scan, views, _ = bound_session((0., 16., -1., 17.), unit=unit)
    assert all(v.geometry is not None and v.geometry.poni is not None for v in views)
    selection = scan.extra['energy_selection']
    assert selection['frame_labels'] == [7, 11, 15, 19]
    for view, energy in zip(views, selection['frame_energies_eV']):
        assert view.geometry.poni['wavelength'] == energy_eV_to_wavelength_m(energy)
    from xrd_tools.geometry_figures import predict_detector_rings
    curves = [predict_detector_rings(v.geometry.poni, 'LaB6', shape=(195, 487)).curves for v in views]
    assert curves[1] and curves[3]
    assert not np.array_equal(curves[1][0].xy, curves[3][0].xy)
    _, _, reference, _ = bound_session((0., 16., -1., 17.), unit=unit, publish=False)
    for actual, before in zip(views, reference):
        np.testing.assert_array_equal(actual.axis_1d.values, before.axis_1d.values)
        np.testing.assert_array_equal(actual.intensity_1d, before.intensity_1d)


def test_gi_raw_rings_use_detector_geometry():
    _, _, first, _ = bound_session(gi=GIMode(incident_angle=.2))
    assert first[0].geometry is not None and first[0].geometry.poni is not None
    _, _, second, _ = bound_session(gi=GIMode(incident_angle=.4))
    from xrd_tools.geometry_figures import predict_detector_rings
    a = predict_detector_rings(first[0].geometry.poni, 'LaB6', shape=(195, 487))
    b = predict_detector_rings(second[0].geometry.poni, 'LaB6', shape=(195, 487))
    for x, y in zip(a.curves, b.curves):
        np.testing.assert_array_equal(x.xy, y.xy)
    assert a.x_label.lower().find('column') >= 0 and a.units == 'px'


@pytest.mark.parametrize('case', ('identity', 'quarter', 'tilted', 'offbeam', 'upstream', 'straddling', 'wrong_orientation'))
def test_analytic_rings_match_pyfai_without_detector_maps(case, monkeypatch):
    from contourpy import contour_generator
    from scipy.spatial import cKDTree
    from pyFAI.calibrant import get_calibrant
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    import xrd_tools.geometry_figures as gf
    from xrd_tools.core.geometry.camera import DetectorCamera
    from tests.core.test_geometry_figures import inputs, reference
    if case in ('upstream', 'straddling'):
        shape = (101, 151)
        ai = AzimuthalIntegrator(dist=.1, poni1=.0505, poni2=.0755,
             rot2=2*np.pi/3 if case == 'upstream' else np.pi/2,
             detector=Detector(pixel1=.001, pixel2=.001, max_shape=shape), wavelength=1e-10)
    else:
        value = inputs(tilt=(3., -2.) if case == 'tilted' else (0., 0.),
                       pose=(40., 12.) if case == 'offbeam' else (0., 0.))
        camera = DetectorCamera('z-', 'x-') if case == 'quarter' else DetectorCamera('x-', 'z+')
        ai, _ = reference(value, camera)
        shape = value.shape[::-1] if case == 'quarter' else value.shape
    angles = ai.center_array(shape, unit='2th_rad', scale=False)
    calibrant = get_calibrant('LaB6'); calibrant.wavelength = ai.wavelength
    rings = [(i, theta) for i, theta in enumerate(calibrant.get_2th(), 1)
             if theta is not None and np.nanmin(angles) < theta < np.nanmax(angles)]
    assert rings
    contours = {i: contour_generator(z=angles).lines(theta) for i, theta in rings}
    def forbidden(*a, **k):
        raise AssertionError('prediction constructed a detector-sized angle map')
    monkeypatch.setattr(ai, 'center_array', forbidden)
    monkeypatch.setattr(ai, 'array_from_unit', forbidden)
    # Numeric controls use the accepted primitive on the parent; no absent-API red.
    points = getattr(gf, 'cone_plane', gf._cone_plane)(ai)
    sampler = getattr(gf, 'display_arcs', gf._display_arcs)
    for number, theta in rings:
        arcs = tuple(sampler(points, theta, shape, 1.))
        if not contours[number]:
            assert not arcs
            continue
        cloud = np.concatenate(contours[number])
        assert arcs
        predicted = np.concatenate(arcs)
        assert cKDTree(cloud).query(predicted)[0].max() <= .5
        assert cKDTree(predicted).query(cloud)[0].max() <= .5
        residual = np.abs(ai.tth(predicted[:, 1], predicted[:, 0]) - theta)
        assert residual.max() / (ai.detector.pixel1/ai.dist) <= .05
        for arc in arcs:
            assert np.isfinite(arc).all() and (arc >= -1e-8).all()
            assert np.linalg.norm(np.diff(arc, axis=0), axis=1).max() <= .5
        if case == 'wrong_orientation':
            turned = predicted.copy(); turned[:, 0] = shape[1]-1-turned[:, 0]
            if number == rings[0][0]:
                assert cKDTree(cloud).query(turned)[0].max() > .5
    if hasattr(gf, 'predict_detector_rings'):
        from xrd_tools.session.readiness import effective_ring_geometry
        cal = DetectorCalibration(PONI(ai.dist, ai.poni1, ai.poni2, ai.rot1, ai.rot2, ai.rot3,
                                      ai.wavelength, ai.detector.name), ai.detector.get_config())
        snapshot = effective_ring_geometry(cal, integrator=ai)
        # All maps on any newly constructed integrator are forbidden too.
        monkeypatch.setattr(AzimuthalIntegrator, 'center_array', forbidden)
        monkeypatch.setattr(AzimuthalIntegrator, 'array_from_unit', forbidden)
        result = gf.predict_detector_rings(snapshot, 'LaB6', shape=shape, step_px=.4)
        assert result.curves and all(not c.xy.flags.writeable for c in result.curves)
        for curve in result.curves:
            theta = np.radians(curve.two_theta_deg)
            assert (abs(ai.tth(curve.xy[:, 1], curve.xy[:, 0])-theta)/(ai.detector.pixel1/ai.dist)).max() <= .05
        with pytest.raises(ValueError, match='parallax'):
            gf.predict_detector_rings({**snapshot, 'parallax': True}, 'LaB6', shape=shape)
        with pytest.raises(ValueError, match='flat|uniform|undistorted'):
            gf.predict_detector_rings({**snapshot, 'uniform_pixel': False}, 'LaB6', shape=shape)
