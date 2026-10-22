"""Direct headless contracts for the bounded phase overlay job and cache."""
from dataclasses import replace
import hashlib
import threading

import numpy as np
import pytest

from xrd_tools.analysis.phase import HAS_PYMATGEN
from xrd_tools.analysis.phase_references import PhaseEntry, build_phase_references
from tests.core.test_phase_references import _CIF

pytestmark = pytest.mark.skipif(not HAS_PYMATGEN, reason="requires pymatgen")


def entry():
    raw = _CIF.encode()
    return PhaseEntry(hashlib.sha256(raw).hexdigest(), 'Silicon', 'cif', raw)


def test_overlay_floor_reuses_one_reachable_listing_and_cache_clear_releases_it(monkeypatch):
    reference = build_phase_references(entry(), wavelength_angstrom=1., q_range=(1., 5.), min_intensity=0.)
    assert reference.phases and not reference.reason
    from xrd_tools.analysis.phase_overlays import PhaseOverlayJob, PhaseOverlayRequest
    import xrd_tools.analysis.phase_references as provider
    calls, actual = [], provider.build_phase_references
    def observed(*args, **kwargs):
        calls.append(kwargs); return actual(*args, **kwargs)
    monkeypatch.setattr(provider, 'build_phase_references', observed)
    job = PhaseOverlayJob()
    request = PhaseOverlayRequest((entry(),), 5., 'q', (1., 5.), 1.)
    first = job(request, threading.Event(), None)[0]
    second = job(replace(request, floor=70.), threading.Event(), None)[0]
    assert len(calls) == 1 and calls[0]['min_intensity'] == 0.
    assert first.q_lines == tuple(r.q for r in reference.phases[0].reflections if r.intensity >= 5.)
    assert second.q_lines == tuple(r.q for r in reference.phases[0].reflections if r.intensity >= 70.)
    assert first.lines == first.q_lines
    job.clear_cache()
    job(request, threading.Event(), None)
    assert len(calls) == 2


def test_cache_clear_during_real_listing_prevents_a_stale_cache_commit(monkeypatch):
    from xrd_tools.analysis.phase_overlays import PhaseOverlayJob, PhaseOverlayRequest
    import xrd_tools.analysis.phase_references as provider
    entered, release, calls = threading.Event(), threading.Event(), []
    actual = provider.build_phase_references
    def observed(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            entered.set(); assert release.wait(20)
        return actual(*args, **kwargs)
    monkeypatch.setattr(provider, 'build_phase_references', observed)
    job = PhaseOverlayJob()
    request = PhaseOverlayRequest((entry(),), 5., 'q', (1., 5.), 1.)
    result = []
    worker = threading.Thread(target=lambda: result.append(job(request, threading.Event(), None)))
    worker.start()
    try:
        assert entered.wait(5)
        job.clear_cache(); release.set(); worker.join(20)
        assert not worker.is_alive() and result[0][0].q_lines
        job(request, threading.Event(), None)
        assert len(calls) == 2
    finally:
        release.set(); worker.join(20)


def test_floor_overflow_fallback_preserves_filtered_provider_caps(monkeypatch):
    import xrd_tools.analysis.phase_references as provider
    monkeypatch.setattr(provider, 'MAX_REFLECTIONS_PER_PHASE', 2)
    assert 'limit' in build_phase_references(entry(), wavelength_angstrom=1., q_range=(1., 7.), min_intensity=0.).reason
    admitted = build_phase_references(entry(), wavelength_angstrom=1., q_range=(1., 7.), min_intensity=100.)
    assert admitted.phases and not admitted.reason
    from xrd_tools.analysis.phase_overlays import PhaseOverlayJob, PhaseOverlayRequest
    actual, calls = provider.build_phase_references, []
    def observed(*args, **kwargs):
        calls.append(kwargs['min_intensity']); return actual(*args, **kwargs)
    monkeypatch.setattr(provider, 'build_phase_references', observed)
    job = PhaseOverlayJob()
    request = PhaseOverlayRequest((entry(),), 100., 'q', (1., 7.), 1.)
    family, = job(request, threading.Event(), None)
    assert family.q_lines == tuple(r.q for r in admitted.phases[0].reflections)
    assert calls == [0., 100.]
    job(request, threading.Event(), None)
    assert calls == [0., 100.]
    refused, = job(replace(request, floor=0.), threading.Event(), None)
    assert not refused.q_lines and 'limit' in refused.reason


def test_native_phase_arcs_share_p3a_and_keep_line_and_ring_normalization_separate():
    from tests.core.test_predicted_ring_overlays import bound_session
    from xrd_tools.core.containers import PONI
    from xrd_tools.core.geometry.diffractometer import DetectorCalibration
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator
    import json
    import xrd_tools.geometry_figures as geometry
    from xrd_tools.analysis.phase_overlays import PhaseOverlayJob, PhaseOverlayRequest
    _, _, views, _ = bound_session()
    snapshot = views[0].geometry.poni
    ai = detector_calibration_to_integrator(DetectorCalibration(
        PONI.from_dict(snapshot), json.loads(snapshot['detector_config_json'])))
    native = ai.twoThetaArray((195, 487))
    reflections = build_phase_references(entry(), wavelength_angstrom=ai.wavelength*1e10,
        two_theta_range=tuple(np.rad2deg((native.min(), native.max()))), min_intensity=5.).phases[0].reflections
    expected = [arc for r in reflections for arc in geometry.display_arcs(
        geometry.cone_plane(ai), 2.*np.arcsin(r.q*ai.wavelength*1e10/(4.*np.pi)), (195,487), 5.)]
    request = PhaseOverlayRequest((entry(),), 5., 'q', (1., 4.), ai.wavelength*1e10,
                                  tuple(snapshot.items()), (195, 487))
    family, = PhaseOverlayJob()(request, threading.Event(), None)
    assert len(family.rings) == len(expected) and family.rings
    for curve, points in zip(family.rings, expected):
        np.testing.assert_allclose(curve.xy, points, rtol=0., atol=1e-9)
        assert not curve.xy.flags.writeable
    assert family.q_lines and all(1. <= q <= 4. for q in family.q_lines)


def test_axis_conversion_missing_wavelength_and_cancellation_are_explicit():
    from xrd_tools.analysis.phase_overlays import PhaseOverlayJob, PhaseOverlayRequest, positions_for_axis
    qs = (1., 2., 3.)
    assert positions_for_axis(qs, 'q', None).values == qs
    assert 'wavelength' in positions_for_axis(qs, '2theta', None).reason
    np.testing.assert_allclose(positions_for_axis(qs, '2theta', 1.).values,
                               2.*np.rad2deg(np.arcsin(np.asarray(qs)/(4.*np.pi))))
    request = PhaseOverlayRequest((entry(),), 5., 'q', (1., 20.), None)
    family, = PhaseOverlayJob()(request, threading.Event(), None)
    assert family.q_lines and '1.5406' in family.reason and 'wavelength' in family.reason
    cancelled = threading.Event(); cancelled.set()
    with pytest.raises(InterruptedError):
        PhaseOverlayJob()(request, cancelled, None)


def test_one_phase_over_its_cap_does_not_blank_the_others(tmp_path):
    """Only the oversized phase is refused, with its own reason; the other phases keep their lines."""
    from xrd_tools.analysis.phase_references import PhaseLibrary
    from xrd_tools.analysis.phase_overlays import PhaseOverlayJob, PhaseOverlayRequest
    big = PhaseLibrary(tmp_path / "library").add_lattice("big cell", a=13., b=14., c=15., space_group=1)
    alone, = PhaseOverlayJob()(PhaseOverlayRequest((entry(),), 5., 'q', (1., 7.), 1.), threading.Event(), None)
    assert alone.q_lines and not alone.reason
    families = {family.id: family for family in PhaseOverlayJob()(
        PhaseOverlayRequest((entry(), big), 5., 'q', (1., 7.), 1.), threading.Event(), None)}
    assert families[entry().id].q_lines == alone.q_lines and not families[entry().id].reason, (
        "a CIF phase within its caps must keep its lines when another phase is refused")
    assert not families[big.id].q_lines and "limit" in families[big.id].reason
