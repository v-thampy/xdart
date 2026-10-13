"""LIVE-SPEC-METADATA-20261012 frozen headless oracle (Claude, October 13): S1-S9.

A TIFF series whose metadata is one SPEC scan, qualified and re-validated through the shared execution-graph owner.
SPEC files only grow during an acquisition: rows, later scans. Growth must keep an admitted graph valid and
undrifted, a rewritten admitted row or header must not, and nothing new may reach the persisted record. Injected
changes are real file writes. Parent behaviour at 709c933f is in the kickoff's freeze record.
"""
from pathlib import Path

import fabio
import numpy as np
import pytest

import xrd_tools.sources.execution_graph as execution_graph
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.sources.execution_graph import (
    SourceRevisionChanged,
    published_metadata_drift,
    qualify_source_execution_graph,
    source_execution_identity_v1_projection,
    source_snapshots_projection,
    validate_source_execution_graph,
)
from xrd_tools.sources.point_images import PointImagesRequired

HEAD = '#F calibrant\n#E 1\n#O0 th  chi\n\n#S 1 ascan th 0 1 10 1\n#P0 0.5 1\n#N 2\n#L del  i0\n'
LATER_SCAN = '\n#S 2 ascan th 0 1 10 1\n#P0 0.5 1\n#N 2\n#L del  i0\n0.0 200\n0.1 201\n'


def _row(k):
    return f'{k / 10:.1f} {100 + k}\n'


def _acquisition(tmp_path, images, rows, *, sidecars=False):
    """A SPEC file with ``rows`` rows of scan 1 and ``images`` SSRL-named images beside it."""
    raw = tmp_path / 'raw'
    raw.mkdir()
    spec = raw / 'calibrant'
    spec.write_text(HEAD + ''.join(_row(k) for k in range(rows)))
    paths = []
    for k in range(images):
        path = raw / f'calibrant_scan001_{k:04d}.tif'
        fabio.tifimage.TifImage(data=np.full((8, 8), k + 1, dtype=np.uint16)).write(str(path))
        if sidecars:
            path.with_suffix('.txt').write_text(f'# Counters\nI0 = {k + 1}\n# Motors\nenergy = 16\n')
        paths.append(path)
    return raw, spec, paths


def _series(raw, paths, metadata_format='spec'):
    """The TIFF series a directory Run admits for one scan (output_preflight._directory_items)."""
    return SourceSpec(raw, SourceKind.TIFF_SERIES, options={
        'selected_file': str(paths[0]), 'files': tuple(str(path) for path in paths),
        'scan_name': 'calibrant_scan001', 'metadata_format': metadata_format})


def _rewrite(path, old, new):
    assert len(old) == len(new)
    text = path.read_text()
    assert text.count(old) == 1
    path.write_text(text.replace(old, new))


def test_s1_rows_appended_after_an_admitted_graph_keep_it_valid(tmp_path):
    raw, spec, paths = _acquisition(tmp_path, images=2, rows=2)
    graph = qualify_source_execution_graph(_series(raw, paths))
    with spec.open('a') as stream:
        stream.write(_row(2) + _row(3))
    validate_source_execution_graph(graph)


def test_s2_a_later_scan_appended_keeps_an_admitted_graph_valid(tmp_path):
    raw, spec, paths = _acquisition(tmp_path, images=2, rows=2)
    graph = qualify_source_execution_graph(_series(raw, paths))
    with spec.open('a') as stream:
        stream.write(LATER_SCAN)
    validate_source_execution_graph(graph)


def test_s3_a_rewritten_admitted_row_breaks_the_graph(tmp_path):
    raw, spec, paths = _acquisition(tmp_path, images=2, rows=2)
    graph = qualify_source_execution_graph(_series(raw, paths))
    _rewrite(spec, '0.1 101\n', '0.1 109\n')
    with pytest.raises(SourceRevisionChanged):
        validate_source_execution_graph(graph)


def test_s4_a_rewritten_motor_header_breaks_the_graph(tmp_path):
    raw, spec, paths = _acquisition(tmp_path, images=2, rows=2)
    graph = qualify_source_execution_graph(_series(raw, paths))
    _rewrite(spec, '#O0 th  chi\n', '#O0 tth chi\n')
    with pytest.raises(SourceRevisionChanged):
        validate_source_execution_graph(graph)


def test_s5_published_drift_ignores_appended_rows_and_reports_a_rewritten_row(tmp_path):
    raw, spec, paths = _acquisition(tmp_path, images=2, rows=1)
    published = qualify_source_execution_graph(_series(raw, paths[:1]))
    with spec.open('a') as stream:
        stream.write(_row(1) + _row(2))
    grown = qualify_source_execution_graph(_series(raw, paths))
    assert published_metadata_drift(published, grown) == ()
    _rewrite(spec, '0.0 100\n', '0.0 105\n')
    rewritten = qualify_source_execution_graph(_series(raw, paths))
    assert [Path(path).name for path in published_metadata_drift(published, rewritten)] == ['calibrant']


def test_s6_rows_appended_between_members_reads_still_admit_one_spec_state(tmp_path, monkeypatch):
    raw, spec, paths = _acquisition(tmp_path, images=3, rows=3)
    real, appended = execution_graph.SourceFileState.capture, []

    def capture(cls, path):
        if Path(path).suffix == '.tif':  # each image member's capture: the acquisition writes another row
            with spec.open('a') as stream:
                stream.write(_row(3 + len(appended)))
            appended.append(Path(path).name)
        return real(path)
    monkeypatch.setattr(execution_graph.SourceFileState, 'capture', classmethod(capture))
    graph = qualify_source_execution_graph(_series(raw, paths))
    monkeypatch.undo()
    assert len(appended) >= 3, 'rows landed between the members\' reads'
    states = {source.metadata_file for source in graph.stamp.metadata_sources}
    assert len(states) == 1 and Path(next(iter(states)).path).name == 'calibrant', states
    assert [target.roles for target in graph.stamp.canonical_targets
            if Path(target.resolved_path).name == 'calibrant'] == [('image_metadata',)]


def test_s7_nothing_new_reaches_the_persisted_record(tmp_path):
    raw, _spec, paths = _acquisition(tmp_path, images=2, rows=2)
    graph = qualify_source_execution_graph(_series(raw, paths))
    record = graph.stamp.as_dict()
    assert [set(item) for item in record['metadata_sources']] == [{'source_path', 'metadata_file'}] * 2
    assert all(set(item['metadata_file']) == {'path', 'size', 'mtime_ns', 'ctime_ns', 'device', 'inode'}
               for item in record['metadata_sources'])
    identity = source_execution_identity_v1_projection(graph)
    assert set(identity) == {'schema_version', 'aliases', 'targets'}
    assert all(set(target) == {'target_id', 'resolved_path', 'state', 'roles'} for target in identity['targets'])
    keys = {writer: next(set(value) for path, value in source_snapshots_projection(graph, writer=writer).items()
                         if Path(path).name == 'calibrant') for writer in (True, False)}
    assert keys == {True: {'adapter_id', 'size', 'mtime_ns', 'frame_count', 'self_contained'},
                    False: {'adapter_id', 'path', 'size', 'mtime_ns', 'ctime_ns', 'device', 'inode',
                            'frame_count', 'self_contained', 'source_role'}}


@pytest.mark.parametrize('rows, expected', [(5, (0.0, 0.1, 0.2, 0.3)), (3, (0.0, 0.1, 0.2, 0.2))],
                         ids=['own-rows-with-later-rows-and-scans', 'finite-past-the-last-row-keeps-clamping'])
def test_s8_each_image_reads_its_own_row(tmp_path, rows, expected):
    raw, spec, paths = _acquisition(tmp_path, images=4, rows=rows)
    with spec.open('a') as stream:
        stream.write(LATER_SCAN)
    graph = qualify_source_execution_graph(_series(raw, paths), selected_motor='del')
    assert tuple(value.value for value in graph.stamp.admitted_motor_values) == expected


def test_s9_live_qualification_infers_no_point_grouping(tmp_path):
    raw, _spec, paths = _acquisition(tmp_path, images=2, rows=1, sidecars=True)
    with pytest.raises(PointImagesRequired):  # finite qualification keeps its grouping inference
        qualify_source_execution_graph(_series(raw, paths, 'auto'))
    graph = qualify_source_execution_graph(_series(raw, paths, 'auto'), infer_point_images=False)
    assert len(graph.stamp.members) == 2
    validate_source_execution_graph(graph)  # nor does its re-check infer one
