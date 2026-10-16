"""LIVE-REVALIDATION-COST frozen headless oracle (Claude, October 15): C1-C3.

Real SPEC files and a real watched directory, read through the product owners. C1/C2 pin every value the
live and finite SPEC readers return against silx's own per-row, per-label lookup on the same file (the
current extraction), so a table-based extraction keeps first-match duplicate labels and the motor-over-
counter collision precedence. C3 pins the directory session's per-candidate probe semantics that a group
probe path must keep. Parent polarity at 8f4a48a3 is recorded in the task's CHECKPOINT.md.
"""
from pathlib import Path

import fabio
import numpy as np
import pytest
from silx.io.specfile import SpecFile

import xrd_tools.sources.registry  # noqa: F401  # register built-in readers
from xrd_tools.io.metadata import read_image_metadata_observed, read_spec_points
from xrd_tools.sources.directory_index import StaleCandidateError
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xrd_tools.sources.probe import ProbeState

#: 22 columns: a duplicate counter label (first match wins) and a counter named like a motor (motor wins).
LABELS = ('del', 'i0', *(f'c{j}' for j in range(2, 20)), 'i0', 'th')
MOTORS = ('th', 'chi', 'tth')
HEAD = '#F calibrant\n#E 1\n#O0 ' + '  '.join(MOTORS) + '\n\n'


def _scan_text(scan, rows):
    lines = [f'#S {scan} ascan th 0 1 {rows} 1', '#P0 0.5 1.25 2.5', f'#N {len(LABELS)}',
             '#L ' + '  '.join(LABELS)]
    lines += [' '.join(f'{scan * 1e6 + row * 1e3 + col:.1f}' for col in range(len(LABELS)))
              for row in range(rows)]
    return '\n'.join(lines) + '\n'


def _spec_file(tmp_path, *, rows, history_bytes=0):
    """A SPEC file whose selected scan is the last one; with ``history_bytes``, earlier scans come first."""
    raw = tmp_path / 'raw'
    raw.mkdir()
    spec = raw / 'calibrant'
    parts, size, scan = [HEAD], len(HEAD), 1
    while size < history_bytes:
        text = _scan_text(scan, 400)
        parts.append(text)
        size += len(text)
        scan += 1
    parts.append(_scan_text(scan, rows))
    spec.write_text(''.join(parts))
    return raw, spec, scan


def _images(raw, scan, count):
    return tuple(raw / f'calibrant_scan{scan:03d}_{k:04d}.tif' for k in range(count))


def _silx_rows(spec, scan, rows):
    """silx's own per-row, per-label lookup: the reference for every reader."""
    parser = SpecFile(str(spec))
    try:
        selected = parser[f'{scan}.1']
        motors = {name: float(selected.motor_position_by_name(name)) for name in selected.motor_names}
        return [{**{label: float(selected.data_column_by_name(label)[row]) for label in selected.labels},
                 **motors} for row in rows]
    finally:
        parser.close()


@pytest.mark.parametrize('history_bytes', [0, 20 << 20], ids=['fresh', 'history20mb'])
def test_c1_read_spec_points_returns_silx_per_label_values_live_and_finite(tmp_path, history_bytes):
    raw, spec, scan = _spec_file(tmp_path, rows=60, history_bytes=history_bytes)
    images = _images(raw, scan, 60)
    expected = _silx_rows(spec, scan, range(60))
    assert expected[0]['i0'] == scan * 1e6 + 1, 'the duplicate label resolves to its first column'
    assert expected[0]['th'] == 0.5, 'the motor wins over the counter of the same name'
    live = read_spec_points(images, published=59)
    finite = read_spec_points(images)
    assert live is not None and finite is not None
    for points in (live, finite):
        assert points.path == spec
        assert [dict(read.values) for read in points.reads] == expected
        assert all(read.source_path == spec for read in points.reads)
    assert live.extent == finite.extent and len(live.extent.marks) == 61


@pytest.mark.parametrize('max_input_bytes', [None, 1 << 26], ids=['live_reader', 'bounded_snapshot'])
def test_c2_the_per_frame_spec_reader_returns_silx_per_label_values_and_clamps(tmp_path, max_input_bytes):
    raw, spec, scan = _spec_file(tmp_path, rows=40, history_bytes=2 << 20)
    for k in (0, 17, 39, 45):
        image = raw / f'calibrant_scan{scan:03d}_{k:04d}.tif'
        read = read_image_metadata_observed(image, 'spec', max_input_bytes=max_input_bytes)
        assert read.source_path == spec
        assert dict(read.values) == _silx_rows(spec, scan, [min(k, 39)])[0], k


def _tiff(path, value):
    fabio.tifimage.TifImage(data=np.full((4, 4), value, dtype=np.uint16)).write(str(path))


def test_c3_session_candidate_probes_keep_cached_stale_and_refresh_semantics(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    for k in range(12):
        _tiff(raw / f'scan_{k:04d}.tif', k)
    session = DirectoryIndexSession(retry_deadline=float('inf'), probe_candidates=False)
    try:
        session.configure(raw, suffixes=('.tif',))
        candidates = session.observe(refresh=True).discovered_snapshot.candidates
        first = [session.probe_candidate(value, refresh=False) for value in candidates]
        assert [value.candidate for value in first] == list(candidates)
        assert all(value.result.state is ProbeState.READY for value in first)
        again = [session.probe_candidate(value, refresh=False) for value in candidates]
        assert [(value.candidate, value.result) for value in again] == [
            (value.candidate, value.result) for value in first]
        changed = candidates[5]
        fabio.tifimage.TifImage(data=np.full((4, 8), 9, dtype=np.uint16)).write(str(changed.path))
        with pytest.raises(StaleCandidateError):
            session.probe_candidate(changed, refresh=True)
        removed = candidates[7]
        removed.path.unlink()
        refreshed = {value.path: value for value in session.observe(refresh=True).discovered_snapshot.candidates}
        assert removed.path not in refreshed
        with pytest.raises(StaleCandidateError):
            session.probe_candidate(removed, refresh=False)
        current = refreshed[changed.path]
        assert current != changed
        reprobed = session.probe_candidate(current, refresh=False)
        assert reprobed.candidate == current and reprobed.result.state is ProbeState.READY
        unchanged = [session.probe_candidate(value, refresh=False) for value in candidates[:5]]
        assert [value.result for value in unchanged] == [value.result for value in first[:5]]
    finally:
        session.close()
