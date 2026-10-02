from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io import open_scan
from xrd_tools.notebooks.int_replay import run_int_replay
from xrd_tools.session.run_configuration import RunIntent, ThresholdIntent
from xrd_tools.session.run_intent_profile import dump_run_intent_profile
from xrd_tools.sources.selection import DirectorySourceSpec


def _fixture(root):
    root.mkdir(exist_ok=True)
    poni = root / 'cal.poni'
    poni.write_text('poni_version: 2\nDetector: Pilatus100k\nDetector_config: {}\nDistance: 0.1234\nPoni1: 0.01\nPoni2: 0.03\nRot1: 0\nRot2: 0\nRot3: 0\nWavelength: 1e-10\n')
    source = root / 'raw.h5'
    with h5py.File(source, 'w') as handle:
        detector = handle.create_group('entry/instrument/detector')
        detector.create_dataset('data', data=np.full((2, 195, 487), 100, np.uint16))
        handle.create_dataset('entry/measurement/I0', data=[2., 4.])
    return RunIntent(source_spec=SourceSpec(source, SourceKind.NEXUS_STACK),
        project_root=str(root), poni_file=str(poni), save_path=str(root / 'old'),
        max_cores=2, processing_mode='Int 2D',
        bai_1d_args={'npt': 32, 'method': 'histogram', 'unit': 'q_A^-1'},
        bai_2d_args={'npt_rad': 32, 'npt_azim': 12, 'method': 'histogram', 'unit': 'q_A^-1'})


@pytest.mark.parametrize('mode', ['Int 1D', 'Int 2D'])
def test_int_replay_uses_the_same_admitted_assets_and_native_reducer(tmp_path, mode):
    from dataclasses import replace
    from xdart.gui.tabs.scattering.output_preflight import _load_scientific_assets, native_int_reduction_plan
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator
    from xrd_tools.reduction import NexusSink, run_reduction
    from xrd_tools.sources import open_source
    intent = _fixture(tmp_path / 'project')
    intent.processing_mode = mode
    shape = (195, 487)
    for name, array in (('dark', np.full(shape, 3.1)), ('flat', np.full(shape, 2.3)),
                        ('mask', np.zeros(shape, bool))):
        path = tmp_path / 'project' / (name + '.npy')
        np.save(path, array)
        setattr(intent, name + '_file', str(path))
    intent.threshold = ThresholdIntent(apply_threshold=True, threshold_min=0, threshold_max=200)
    assets = _load_scientific_assets(intent)
    configuration = intent.clone_candidate().freeze(generation=1)
    plan = replace(native_int_reduction_plan(configuration, companion_modes_2d=True),
                   mask=assets.mask, dark=assets.dark, flat=assets.flat)
    source = open_source(intent.source_spec)
    baseline = tmp_path / 'baseline.nexus'
    try:
        scan = source.to_scan(poni=assets.poni,
            integrator=detector_calibration_to_integrator(assets.detector_calibration))
        run_reduction(plan, scan, NexusSink(baseline, create_new=True), executor=2)
    finally:
        source.close()
    raw_hash = hashlib.sha256(Path(intent.source_spec.uri).read_bytes()).hexdigest()
    encoded = json.loads(dump_run_intent_profile(intent))
    outputs = run_int_replay(encoded, data_root=intent.project_root,
                             output_root=Path(intent.project_root) / 'replay')
    assert len(outputs) == 1
    expected, actual = open_scan(baseline), open_scan(outputs[0])
    for label in expected.frame_indices:
        first, second = expected.get_1d(label), actual.get_1d(label)
        np.testing.assert_array_equal(first.q, second.q)
        np.testing.assert_array_equal(first.intensity, second.intensity)
        if mode == 'Int 2D':
            np.testing.assert_array_equal(expected.get_2d(label).intensity, actual.get_2d(label).intensity)
    assert hashlib.sha256(Path(intent.source_spec.uri).read_bytes()).hexdigest() == raw_hash
    assert not (Path(intent.project_root) / 'old').exists()
    before = outputs[0].read_bytes()
    with pytest.raises(FileExistsError):
        run_int_replay(encoded, data_root=intent.project_root,
                       output_root=Path(intent.project_root) / 'replay')
    assert outputs[0].read_bytes() == before


def test_int_replay_live_refusal_and_relocated_project(tmp_path):
    import shutil
    intent = _fixture(tmp_path / 'original')
    encoded = json.loads(dump_run_intent_profile(intent))
    relocated = tmp_path / 'relocated'
    shutil.copytree(intent.project_root, relocated)
    outputs = run_int_replay(encoded, data_root=relocated, output_root=relocated / 'replay')
    assert len(outputs) == 1 and outputs[0].is_file()
    assert not (Path(intent.project_root) / 'replay').exists()
    intent.live_mode = True
    with pytest.raises(ValueError, match='LIVE_REPLAY_UNAVAILABLE'):
        run_int_replay(json.loads(dump_run_intent_profile(intent)), data_root=relocated,
                       output_root=relocated / 'live')
    assert not (relocated / 'live').exists()


def test_int_replay_directory_uses_native_scan_grouping(tmp_path):
    intent = _fixture(tmp_path / 'project')
    raw = tmp_path / 'project' / 'inputs'
    raw.mkdir()
    import shutil
    shutil.copyfile(intent.source_spec.uri, raw / 'scan1.h5')
    shutil.copyfile(intent.source_spec.uri, raw / 'scan2.h5')
    intent.source_spec = DirectorySourceSpec(raw, suffixes=('.h5',))
    outputs = run_int_replay(json.loads(dump_run_intent_profile(intent)),
                             data_root=intent.project_root,
                             output_root=Path(intent.project_root) / 'directory')
    assert len(outputs) == 2
    assert len(set(outputs)) == 2
    assert all(len(open_scan(path).frame_indices) == 2 for path in outputs)


def test_shared_plan_honors_the_recorded_gi_exit_convention():
    from xrd_tools.session.readiness import native_int_reduction_plan
    from xrd_tools.session.run_configuration import GIIntent
    from xrd_tools.corrections.grazing import LEGACY_GI_EXIT_ANGLE_CONVENTION
    intent = RunIntent(gi=GIIntent(enabled=True, incidence_motor='Manual',
                     gi_exit_angle_convention=LEGACY_GI_EXIT_ANGLE_CONVENTION))
    assert native_int_reduction_plan(intent.freeze()).gi.gi_exit_angle_convention == LEGACY_GI_EXIT_ANGLE_CONVENTION


@pytest.mark.parametrize('mode', ['Int 1D', 'Int 1D (XYE)'])
def test_int_replay_background_uses_the_shared_submission_owner(tmp_path, mode):
    from xrd_tools.reduction import FrameBackgroundPlan
    from fabio.tifimage import TifImage
    intent = _fixture(tmp_path / 'project')
    intent.processing_mode = mode
    path = Path(intent.project_root) / 'background.tif'
    TifImage(data=np.full((195, 487), 10, np.uint16)).write(str(path))
    intent.background = FrameBackgroundPlan(mode='Single BG File', locator=str(path), scale=2.0)
    encoded = json.loads(dump_run_intent_profile(intent))
    outputs = run_int_replay(encoded, data_root=intent.project_root,
                             output_root=Path(intent.project_root) / 'background-replay')
    intent.background = FrameBackgroundPlan()
    reference = run_int_replay(json.loads(dump_run_intent_profile(intent)),
                              data_root=intent.project_root,
                              output_root=Path(intent.project_root) / 'without-background')
    if mode == 'Int 1D (XYE)':
        assert len(outputs) == 2 and all(path.name.startswith('iq_') for path in outputs)
        actual = np.loadtxt(outputs[0])[:, 1]
        expected = np.loadtxt(reference[0])[:, 1]
    else:
        actual = open_scan(outputs[0]).get_1d(0).intensity
        expected = open_scan(reference[0]).get_1d(0).intensity
    np.testing.assert_allclose(actual, expected * .8, rtol=2e-7)
