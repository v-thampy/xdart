"""HELP-RESIDUALS-20261011 frozen headless oracle (Claude, October 11): HA1, HA3, HA5, HA6, HC1, HC2, HC3, HC4,
HC7, HE4 and HE5.

The kickoff (xdart_docs handoffs/help_residuals_2026-10-11.md) owns the contract. Every row drives the real owner,
and new names are reached inside test bodies after a behavioural assertion, so nothing fails at collection.
"""
from dataclasses import replace
import hashlib
import json
import math
import subprocess
import sys
import textwrap
from types import MappingProxyType

import numpy as np
import pytest


def test_ha1_al2o3_reaches_pyfai_as_alpha_al2o3_at_every_binding_site(tmp_path):
    from pyFAI.calibrant import get_calibrant
    from xrd_tools import geometry_figures as gf
    from xrd_tools.integrate import stitch_autocalibration as autocal
    from xrd_tools.integrate.calibration_seed import predict_frame_seed
    from tests.core.test_calibration_seed import _frame, _record
    reference = get_calibrant('alpha_Al2O3')
    # the seed: Calibrate's pyFAI head start and the frame table's expected rings
    seed = predict_frame_seed(_record(tmp_path), _frame({'del': 20., 'nu': 0.}), calibrant='Al2O3')
    assert seed.rings, 'alpha-Al2O3 rings fall on this frame'
    for ring in seed.rings:
        assert ring.d_spacing_A == pytest.approx(reference.dspacing[ring.number - 1])
    # Refine's calibrant
    assert list(autocal._calibrant('Al2O3', 1e-10).dspacing) == list(reference.dspacing)
    # the orientation display and the teaching example draw the chosen calibrant's rings
    value = replace(gf.GeometryInputs(shape=(195, 487), detector='Pilatus100k'), calibrant='Al2O3')
    camera = gf.camera_candidates()[0].camera
    shown = gf.orientation_display(gf.OrientationFrame(np.zeros(value.shape), value), camera)
    assert shown.curves, 'the display predicts alpha-Al2O3 rings'
    assert gf._teaching_example(value) is not None
    # the detector-ring kernel
    assert gf.predict_detector_rings(_snapshot('Pilatus100k'), 'Al2O3', shape=(195, 487)).curves
    # the one resolver beside CALIBRANT_NAMES
    from xrd_tools.integrate.calibration import CALIBRANT_NAMES, pyfai_calibrant, pyfai_calibrant_name
    assert pyfai_calibrant_name('Al2O3') == 'alpha_Al2O3'
    assert all(len(pyfai_calibrant(name).dspacing) for name in CALIBRANT_NAMES)
    assert pyfai_calibrant_name('NIST-SRM-660c') == 'NIST-SRM-660c', 'a typed pyFAI name passes through'


_STRUCTURAL_EDIT = textwrap.dedent("""
    import json
    from dataclasses import replace
    import xrd_tools.help_topics as topics
    edited = []
    for topic in topics.TOPICS:
        if topic.identifier == 'calibration.multi-step':
            text = topic.text.replace('\\n\\nUnits and conventions', '\\n\\nConventions')
            text = text.replace('4. Calibrate each', 'A note on frames.\\n\\n4. Calibrate each')
            assert text != topic.text
            topic = replace(topic, text=text)
        edited.append(topic)
    topics.TOPICS = tuple(edited)
    from xrd_tools.integrate.calibration_checklist import ChecklistFacts, calibration_checklist
    try:
        rows = calibration_checklist(ChecklistFacts())
    except Exception as error:
        print(json.dumps({'error': type(error).__name__ + ': ' + str(error)}))
    else:
        print(json.dumps({'rows': [[row.step, row.text] for row in rows]}))
""")


def test_ha3_a_prose_edit_cannot_break_the_checklist(tmp_path):
    completed = subprocess.run([sys.executable, '-c', _STRUCTURAL_EDIT], capture_output=True, text=True,
                               timeout=120, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr[-2000:]
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert 'error' not in payload, payload
    assert len(payload['rows']) == 8, payload
    from xrd_tools.help_topics import CALIBRATION_STEPS, get_topic
    from xrd_tools.integrate.calibration_checklist import ChecklistFacts, calibration_checklist
    rows = calibration_checklist(ChecklistFacts())
    assert [row.text for row in rows[:7]] == [text for _label, text in CALIBRATION_STEPS]
    assert [row.step for row in rows[:7]] == [label for label, _text in CALIBRATION_STEPS]
    prose = ' '.join(get_topic('calibration.multi-step').text.split())
    assert all(text in prose for _label, text in CALIBRATION_STEPS), 'the topic lists the same seven steps'
    assert [row[1] for row in payload['rows'][:7]] == [text for _label, text in CALIBRATION_STEPS]


def test_ha5_step_four_names_the_three_frame_minimum():
    from xrd_tools.integrate.calibration_checklist import ChecklistFacts, calibration_checklist
    two = ChecklistFacts(frame_count=2, complete_count=2, frame_message='2 of 2 Complete')
    row = calibration_checklist(two)[3]
    assert row.state == 'Needs input'
    assert '2 of 2 Complete' in row.message and 'three' in row.message.lower(), row.message
    three = calibration_checklist(replace(two, frame_count=3, complete_count=3, frame_message='3 of 3 Complete'))[3]
    assert three.state == 'Done' and 'three' not in three.message.lower(), three.message


def _calibrant_image(value, camera, calibrant, senses=(1, 1)):
    """tests/core/test_geometry_figures.py's ``reference`` with the calibrant chosen: an independent upstream
    pyFAI image for ``camera`` at ``value``'s fixed geometry."""
    from pyFAI.calibrant import get_calibrant
    from xrd_tools.core.energy import wavelength_m_to_energy_eV
    from xrd_tools.core.geometry.camera import stitch_orientation
    from xrd_tools.integrate.calibration_seed import predict_frame_seed, rough_arm_record
    from xrd_tools.integrate.stitch_calibration import CalibrationFrame
    from tests.core.test_geometry_figures import PITCH, SIGNAL
    orientation = stitch_orientation(camera)
    oriented_shape = value.shape[::-1] if orientation.swaps_axes else value.shape
    beam = (np.array(oriented_shape)[::-1] - 1) / 2 + value.beam_offset_px
    record = rough_arm_record(detector=value.detector, distance_m=value.distance_m,
                              wavelength_m=value.wavelength_m, beam_px=beam, camera=orientation,
                              mount_deg=value.mount_deg, orientation=value.first_pixel,
                              senses=dict(zip(('del', 'nu'), senses)))
    frame = CalibrationFrame('f', 's', 'spec', {}, 0, 'raw', 0, 'i', 'sha', oriented_shape,
                             0, dict(value.readings), wavelength_m_to_energy_eV(value.wavelength_m))
    seed = predict_frame_seed(record, frame)
    reference = get_calibrant(calibrant)
    reference.wavelength = value.wavelength_m
    image = reference.fake_calibration_image(seed.integrator, shape=oriented_shape, Imax=SIGNAL,
                                             resolution=np.degrees(PITCH / value.distance_m))
    return orientation.unapply(image)


def test_ha6_orientation_scores_the_chosen_calibrants_rings():
    from xrd_tools import geometry_figures as gf
    from xrd_tools.core.geometry.camera import DetectorCamera
    from tests.core.test_geometry_figures import inputs
    truth = DetectorCamera('x-', 'z+')
    frames = []
    for pose in ((0., 0.), (15., 0.), (0., 12.)):
        value = replace(inputs(pose=pose), calibrant='CeO2')
        frames.append(gf.OrientationFrame(_calibrant_image(value, truth, 'CeO2'), value, None))
    result = gf.diagnose_orientation(tuple(frames))
    true = next(c for c in result.candidates if c.camera == truth and c.senses == (1, 1))
    assert true.ring_scores and true.score > .9, (
        'the true camera measures the CeO2 rings that are in the image', true.score)
    assert result.identifiable and result.best.camera == truth and result.best.senses == (1, 1), result.reason


def _cif_file(root, name, a):
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter
    path = root / f'{name}.cif'
    path.write_text(str(CifWriter(Structure(Lattice.cubic(a), ['Cs', 'Cl'], [[0, 0, 0], [.5, .5, .5]]))))
    return path


def test_hc1_the_chooser_listing_reads_no_cif_bytes(tmp_path, monkeypatch):
    from pathlib import Path
    from xrd_tools.analysis.phase_references import PhaseLibrary
    library = PhaseLibrary(tmp_path / 'library')
    kept = library.add_cif(_cif_file(tmp_path, 'kept', 4.10))
    changed = library.add_cif(_cif_file(tmp_path, 'changed', 4.20))
    (tmp_path / 'library' / (changed.id + '.cif')).write_bytes(b'data_changed\n_cell_length_a 4.2\n')
    opened = []
    real_open = Path.open
    def counted(self, *args, **kwargs):
        if self.suffix == '.cif':
            opened.append(self.name)
        return real_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', counted)
    listing = getattr(library, 'catalogue', library.entries)  # the chooser's listing on either tree
    rows = listing()
    assert not opened, f'listing read CIF bytes: {opened}'
    names = {(row[1] if isinstance(row, tuple) else row.name) for row in rows}
    assert names == {'kept', 'changed'}, names
    monkeypatch.setattr(Path, 'open', real_open)
    assert 'CIF bytes changed' in library.resolve(changed.id).reason
    assert library.resolve(kept.id).entry is not None


def test_hc2_a_broad_peak_cif_fit_and_the_floors_reflections():
    from xrd_tools.analysis.fitting.phase_fitting import PhaseFitter
    from xrd_tools.analysis.phase_references import phase_model
    from tests.core.test_reference_phase_sources import CSCL_WEAK_Q, _entry
    entry = _entry('CsCl')
    truth = phase_model(entry, wavelength_angstrom=1., q_range=(1., 6.))
    a0 = truth.lattice.a
    q = np.linspace(1., 6., 5001)
    y = 20. + sum(1000. * p.intensity / 100. * np.exp(-(q - p.q) ** 2 / (2 * .1 ** 2)) for p in truth.peaks)
    for floor, weak_modelled in ((5., False), (1., True)):
        start = phase_model(entry, wavelength_angstrom=1., q_range=(1., 6.))
        start.update_lattice(a=a0 * 1.004, b=a0 * 1.004, c=a0 * 1.004)  # a cubic start: every edge
        fitter = PhaseFitter(q, y)
        fitter.add_phase(start, q_range=(1., 6.), min_intensity=floor)
        modelled = np.concatenate([np.asarray(arr, float).ravel() for arr in fitter._peak_q_arrays])
        for weak in CSCL_WEAK_Q:  # the start cell is 0.4 % larger, so each listed line sits at q / 1.004
            assert bool(np.any(np.abs(modelled - weak / 1.004) < 2e-3)) is weak_modelled, (floor, weak)
        if weak_modelled:  # every reachable reflection modelled: the broad-peak cell is recovered
            result = fitter.fit(phase_profile='gaussian', texture='none')
            residual = np.sqrt(np.sum((y - fitter.eval_model(result.params)) ** 2) / np.sum(y ** 2))
            assert residual < .06 and abs(result.params['p0_a'].value - a0) < 1.5e-3, (
                residual, result.params['p0_a'].value - a0)


def test_hc3_grid_reasons_name_the_grid_and_ring_reasons_are_unchanged():
    from xrd_tools import geometry_figures as gf
    snapshot = _snapshot('Pilatus100k')
    for case in (None, {**snapshot, 'parallax': True}, {**snapshot, 'uniform_pixel': False}):
        shape = None if case is None else (195, 487)
        reason = gf.detector_grid_unavailable(case, shape)
        assert reason and 'grid' in reason.lower() and 'ring' not in reason.lower(), reason
        assert 'ring' in gf.detector_ring_unavailable(case, shape).lower()


def test_hc4_an_edited_lattice_document_is_refused(tmp_path):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    library = PhaseLibrary(tmp_path / 'library')
    entry = library.add_lattice('cell', a=4.1, space_group=221)
    path = tmp_path / 'library' / (entry.id + '.json')
    document = json.loads(path.read_text())
    document['lattice']['a'] = 5.0  # edited behind its unchanged id
    path.write_text(json.dumps(document, sort_keys=True))
    lookup = library.resolve(entry.id)
    assert lookup.entry is None and 'identity' in lookup.reason.lower(), lookup


def test_hc7_a_merged_line_keeps_every_hkl_family():
    from xrd_tools.analysis.phase_references import build_phase_references
    from tests.core.test_reference_phase_sources import _entry
    listed = build_phase_references(_entry('CsCl'), wavelength_angstrom=1., q_range=(1., 5.), min_intensity=0.)
    assert not listed.reason, listed.reason
    reflections = listed.phases[0].reflections
    merged = min(reflections, key=lambda r: abs(r.q - 2 * math.pi * 3 / 4.12))  # (300) and (221) coincide
    families = getattr(merged, 'hkls', (merged.hkl,))
    assert {tuple(sorted(map(abs, hkl))) for hkl in families} == {(0, 0, 3), (1, 2, 2)}, families
    assert merged.hkl == families[0]
    from xrd_tools.analysis.phase import PhaseModel
    lattice = PhaseModel.from_lattice('cell', a=4.12, space_group='Pm-3m')
    shared = [p for p in lattice.peaks if abs(p.q - 2 * math.pi * 3 / 4.12) < 1e-6]
    assert len(shared) == 1 and len(getattr(shared[0], 'hkls', (shared[0].hkl,))) == 2, shared


def test_he4_frame_geometry_never_shares_a_producers_live_mapping():
    from xrd_tools.core.scan import FrameGeometry
    live = {'dist': .1, 'poni1': .05, 'poni2': .06, 'rot1': 0., 'rot2': 0., 'rot3': 0., 'wavelength': 1e-10}
    geometry = FrameGeometry(rot1=0., rot2=0., rot3=0., poni=MappingProxyType(live))
    live['dist'] = 9.
    assert geometry.poni['dist'] == .1


#: The finite admitted set: Run fixtures' Pilatus100k, the orientation default Pilatus1M, every detector named by
#: the test data's PONI files, and a generic Detector whose pixel size lives only in its configuration.
_ADMITTED = ('Pilatus100k', 'Pilatus1M', 'Pilatus300kw', 'PilatusCdTe300kw', 'Eiger4M', 'Eiger2_500k',
             'RayonixMx225', 'Perkin', 'generic')


def _integrator(name):
    from pyFAI.detectors import Detector, detector_factory
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    detector = (Detector(pixel1=75e-6, pixel2=100e-6, max_shape=(64, 96)) if name == 'generic'
                else detector_factory(name))
    return AzimuthalIntegrator(dist=.2, poni1=.01, poni2=.012, detector=detector, wavelength=1e-10)


def _snapshot(name):
    from xrd_tools.core.containers import PONI
    from xrd_tools.core.geometry.diffractometer import DetectorCalibration
    from xrd_tools.session.readiness import effective_ring_geometry
    ai = _integrator(name)
    calibration = DetectorCalibration(PONI(ai.dist, ai.poni1, ai.poni2, ai.rot1, ai.rot2, ai.rot3,
                                           ai.wavelength, ai.detector.name), ai.detector.get_config())
    return dict(effective_ring_geometry(calibration, integrator=ai))


@pytest.mark.parametrize('name', _ADMITTED)
def test_he5_each_admitted_detector_round_trips_through_the_ring_snapshot(name):
    from xrd_tools.core.containers import PONI
    from xrd_tools.core.geometry.diffractometer import DetectorCalibration
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator
    original = _integrator(name).detector
    snapshot = _snapshot(name)
    rebuilt = detector_calibration_to_integrator(DetectorCalibration(PONI.from_dict(snapshot),
        json.loads(snapshot['detector_config_json']), parallax=snapshot.get('parallax'))).detector
    assert tuple(rebuilt.max_shape) == tuple(original.max_shape), name
    assert (rebuilt.pixel1, rebuilt.pixel2) == pytest.approx((original.pixel1, original.pixel2), abs=0), name
