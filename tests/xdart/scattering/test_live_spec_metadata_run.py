"""LIVE-SPEC-METADATA-20261012 frozen live-Run oracle (Claude, October 13): L1-L8.

Real live Runs: a StandardRunExecutor watches a directory of SSRL-named images whose metadata is a SPEC file that
the acquisition appends to, a row per point, the row landing before or after its image. Every row keeps every
drained event, waits for its own predicate, ends with an explicit Stop, and reads the durable output only after the
Run closed: frame labels and each frame's recorded counter. Parent behaviour at 709c933f is in the kickoff's freeze
record.
"""
import time

import fabio
import h5py
import numpy as np

import xrd_tools.sources.registry  # noqa: F401  # register built-in readers
from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_live_qualify_refusal_run import (
    SIDECARS, _labels, _said, _settle, _stop, _terminal, _wait,
)
from tests.xdart.scattering.test_p1b_output_graph import _start
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xrd_tools.io.nexus import read_scan_metadata
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec

HEAD = '#F calibrant\n#E 1\n#O0 th  chi\n\n'
STEM = 'calibrant_scan{scan:03d}'


def _scan(spec, scan):
    with spec.open('a') as stream:
        stream.write(f'#S {scan} ascan th 0 1 10 1\n#P0 0.5 1\n#N 2\n#L del  i0\n')


def _row(spec, scan, k):
    with spec.open('a') as stream:
        stream.write(f'{k / 10:.1f} {100 * scan + k}\n')


def _image(raw, scan, k, *, sidecar=False):
    """One detector image, renamed into place whole; with ``sidecar``, its .txt sidecar first (I0 = 100 * scan + k)."""
    target = raw / f'{STEM.format(scan=scan)}_{k:04d}.tif'
    if sidecar:
        target.with_suffix('.txt').write_text(SIDECARS['txt'](float(100 * scan + k)))
    partial = target.with_suffix('.part')
    fabio.tifimage.TifImage(data=np.full((195, 487), k + 1, dtype=np.uint16)).write(str(partial))
    partial.replace(target)


def _acquisition(tmp_path, points=1):
    """Scan 1 of a SPEC file with ``points`` complete points (row, then image)."""
    raw = tmp_path / 'raw'
    raw.mkdir()
    spec = raw / 'calibrant'
    spec.write_text(HEAD)
    _scan(spec, 1)
    for k in range(points):
        _row(spec, 1, k)
        _image(raw, 1, k)
    return raw, spec


def _run(tmp_path, raw, request_value, fmt='spec'):
    poni = tmp_path / 'cal.poni'
    write_poni(poni)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=('.tif',), metadata_format=fmt), poni_file=str(poni),
        project_root=str(raw), save_path=str(tmp_path / 'processed'), output_mode='Overwrite',
        processing_mode='Int 1D', live_mode=True, max_cores=1, bai_1d_args={'npt': 8, 'method': 'numpy'}),
        request_value=request_value)
    return executor, identity


def _reduced(label, scan=1):
    return lambda values: label in _labels(values, STEM.format(scan=scan)) or _terminal(values)


def _output(tmp_path, scan=1, counter='i0'):
    """(frame labels, each frame's recorded counter) of one scan's output; None when none was written."""
    path = tmp_path / 'processed' / f'{STEM.format(scan=scan)}_int1d.nexus'
    if not path.exists():
        return None
    with h5py.File(path, 'r') as handle:
        labels = tuple(int(value) for value in handle['entry/integrated_1d/frame_index'][()])
    data = read_scan_metadata(path)
    values = tuple(float(value) for value in data[counter].values) if counter in data else None
    return labels, values


def _pending(events):
    return [event.detail for event in events if event.kind is StandardEventKind.DISCOVERY]


def test_l1_an_image_landing_before_its_spec_row_waits_for_the_row(tmp_path):
    raw, spec = _acquisition(tmp_path)
    executor, identity = _run(tmp_path, raw, 5101)
    events = []
    try:
        _wait(executor, events, _reduced(1), 20.)
        for k in (1, 2):
            _image(raw, 1, k)
            time.sleep(.3)
            _row(spec, 1, k)  # its row, 0.3 s after the image
            _wait(executor, events, _reduced(k + 1), 20.)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _output(tmp_path) == ((1, 2, 3), (100.0, 101.0, 102.0))


def test_l2_a_spec_row_landing_before_its_image_does_not_stop_the_output(tmp_path):
    raw, spec = _acquisition(tmp_path)
    executor, identity = _run(tmp_path, raw, 5102)
    events = []
    try:
        _wait(executor, events, _reduced(1), 20.)
        for k in (1, 2):
            _row(spec, 1, k)
            time.sleep(.3)
            _image(raw, 1, k)  # its image, 0.3 s after the row
            _wait(executor, events, _reduced(k + 1), 20.)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert not _said(events, 'stopped growing'), terminal
    assert _output(tmp_path) == ((1, 2, 3), (100.0, 101.0, 102.0))


def test_l3_an_image_is_never_reduced_with_another_points_row(tmp_path):
    raw, spec = _acquisition(tmp_path)
    executor, identity = _run(tmp_path, raw, 5103)
    events = []
    try:
        _wait(executor, events, _reduced(1), 20.)
        _image(raw, 1, 1)
        _image(raw, 1, 2)
        _settle(executor, events, 1.)  # ten polls with both rows missing
        early, waiting = list(_labels(events, STEM.format(scan=1))), _pending(events)
        _row(spec, 1, 1)
        _row(spec, 1, 2)
        _wait(executor, events, _reduced(3), 20.)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert early == [1], 'neither image was reduced before its own row landed'
    assert waiting and '2 pending' in waiting[-1], waiting
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _output(tmp_path) == ((1, 2, 3), (100.0, 101.0, 102.0))


def test_l4_a_rewritten_published_row_stops_that_output_and_keeps_its_frames(tmp_path):
    raw, spec = _acquisition(tmp_path, points=2)
    executor, identity = _run(tmp_path, raw, 5104)
    events = []
    try:
        _wait(executor, events, _reduced(2), 20.)
        _settle(executor, events)
        text = spec.read_text()
        spec.write_text(text.replace('0.0 100\n', '0.0 105\n'))  # point 1's row rewritten in place
        _wait(executor, events, lambda values: _said(values, 'stopped growing') or _terminal(values), 20.)
        _row(spec, 1, 2)
        _image(raw, 1, 2)
        _settle(executor, events, 1.)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert f'{STEM.format(scan=1)}_int1d.nexus stopped growing' in terminal[0][1], terminal
    assert _output(tmp_path) == ((1, 2), (100.0, 101.0))


def test_l5_a_later_scan_in_the_same_spec_file_leaves_an_earlier_output_alone(tmp_path):
    raw, spec = _acquisition(tmp_path, points=2)
    executor, identity = _run(tmp_path, raw, 5105)
    events = []
    try:
        _wait(executor, events, _reduced(2), 20.)
        _settle(executor, events)
        with spec.open('a') as stream:
            stream.write('\n')
        _scan(spec, 2)
        for k in (0, 1):
            _row(spec, 2, k)
            time.sleep(.3)
            _image(raw, 2, k)
            _wait(executor, events, _reduced(k + 1, scan=2), 20.)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert not _said(events, 'stopped growing'), terminal
    assert _output(tmp_path, 1) == ((1, 2), (100.0, 101.0))
    assert _output(tmp_path, 2) == ((1, 2), (200.0, 201.0))


def test_l6_under_auto_a_spec_file_beside_the_images_infers_no_point_grouping(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    spec = raw / 'calibrant'
    spec.write_text(HEAD)
    _scan(spec, 1)
    _row(spec, 1, 0)
    _image(raw, 1, 0, sidecar=True)
    executor, identity = _run(tmp_path, raw, 5106, fmt='auto')
    events = []
    try:
        _wait(executor, events, _reduced(1), 20.)
        for k in (1, 2):
            _image(raw, 1, k, sidecar=True)  # image and sidecar; the SPEC row 0.3 s later
            time.sleep(.3)
            _row(spec, 1, k)
            _wait(executor, events, _reduced(k + 1), 20.)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _output(tmp_path, counter='I0') == ((1, 2, 3), (100.0, 101.0, 102.0))


def test_l7_a_fast_spec_acquisition_publishes_every_point(tmp_path):
    raw, spec = _acquisition(tmp_path)
    executor, identity = _run(tmp_path, raw, 5107)
    events = []
    try:
        _wait(executor, events, _reduced(1), 20.)
        for k in range(1, 21):  # a point every 30 ms, faster than a poll: its row, then its image
            _row(spec, 1, k)
            _image(raw, 1, k)
            time.sleep(.03)
        _wait(executor, events, _reduced(21), 30.)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert not _said(events, 'stopped growing'), terminal
    assert _output(tmp_path) == (tuple(range(1, 22)), tuple(float(100 + k) for k in range(21)))


def test_l8_an_image_without_its_spec_file_waits_for_it(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _image(raw, 1, 0)  # the SPEC file is not there yet
    executor, identity = _run(tmp_path, raw, 5108)
    events = []
    try:
        _settle(executor, events, 1.)
        early, waiting = list(_labels(events, STEM.format(scan=1))), _pending(events)
        spec = raw / 'calibrant'
        spec.write_text(HEAD)
        _scan(spec, 1)
        _row(spec, 1, 0)
        _wait(executor, events, _reduced(1), 20.)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert early == [], 'nothing was reduced before its SPEC row existed'
    assert waiting and '1 pending' in waiting[-1], waiting
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _output(tmp_path) == ((1,), (100.0,))
