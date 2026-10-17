"""LIVE-SPEC-METADATA-20261012 oracle completion (Claude, October 13, from Codex's pre-writer review): C1-C4.

Strict own-row custody at the headless qualification boundary (CR1) and a SPEC file truncated during the read (CR6).
A live qualification (``infer_point_images=False``) never borrows another point's row and never admits no metadata
for a new member: it waits (a typed ``SourceRevisionChanged``) or refuses a complete line that is not a row (a
``ValueError``). Members the Run already published are admitted without metadata once their rows are gone or invalid,
so the published-metadata fence reports it. Finite qualification keeps today's reads. Injected changes are real file
writes. Parent behaviour is in the kickoff's freeze record.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import xrd_tools.io.metadata as metadata
from tests.core.test_live_spec_metadata import HEAD, _acquisition, _rewrite, _series
from xrd_tools.sources.execution_graph import (
    SourceRevisionChanged,
    published_metadata_drift,
    qualify_source_execution_graph,
)


def _live(raw, paths, **kwargs):
    return qualify_source_execution_graph(_series(raw, paths), infer_point_images=False, **kwargs)


@pytest.mark.parametrize('missing', ['row', 'scan', 'file'])
def test_c1_live_qualification_waits_for_a_missing_row_and_never_borrows_one(tmp_path, missing):
    raw, spec, paths = _acquisition(tmp_path, images=3, rows=2 if missing == 'row' else 3)
    if missing == 'scan':
        spec.write_text(HEAD.split('#S ')[0])  # the file header only: scan 1 is not written yet
    elif missing == 'file':
        spec.unlink()
    with pytest.raises(SourceRevisionChanged):
        _live(raw, paths)


@pytest.mark.parametrize('rows', ['0.0 100\n0.1\n', '0.0 100\n0.1 abc\n', '0.0 100\n0.1 101 7\n',
                                  '0.0 100\n0.1\n0.2 102\n'],
                         ids=['short', 'not-a-number', 'extra-column', 'invalid-line-before-a-valid-row'])
def test_c2_a_complete_line_that_is_not_a_row_is_refused(tmp_path, rows):
    raw, spec, paths = _acquisition(tmp_path, images=rows.count('\n'), rows=0)
    with spec.open('a') as stream:
        stream.write(rows)
    with pytest.raises(ValueError) as caught:
        _live(raw, paths)
    assert not isinstance(caught.value, SourceRevisionChanged), caught.value
    message = str(caught.value)
    assert 'calibrant' in message and 'scan 1' in message and 'row 2' in message, message


@pytest.mark.parametrize('change', ['removed', 'invalid'])
def test_c3_published_rows_gone_or_invalid_are_admitted_without_metadata(tmp_path, change):
    raw, spec, paths = _acquisition(tmp_path, images=2, rows=2)
    published = _live(raw, paths)
    if change == 'removed':
        spec.unlink()
    else:
        _rewrite(spec, '0.0 100\n', '0.0 abc\n')
    current = _live(raw, paths, published_members=2)
    assert [source.metadata_file for source in current.stamp.metadata_sources] == [None, None]
    assert {Path(path).name for path in published_metadata_drift(published, current)} == {'calibrant'}


def test_c4_a_spec_file_truncated_during_the_read_leaves_the_process_alive(tmp_path):
    _raw, spec, paths = _acquisition(tmp_path, images=2, rows=2)
    child = textwrap.dedent('''
        import mmap, os, pathlib, sys
        import xrd_tools.io.metadata as metadata
        spec, images = sys.argv[1], sys.argv[2:]
        real_mmap, real_open = mmap.mmap, pathlib.Path.open

        def same(path):
            return os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(spec))

        def mapped(fileno, length, *args, **kwargs):  # a mapped view: the file is truncated under it
            view = real_mmap(fileno, length, *args, **kwargs)
            os.truncate(spec, 0)
            return view

        class Truncated:  # a bytes read: the file is truncated just before it
            def __init__(self, stream):
                self.stream = stream
            def __getattr__(self, name):
                return getattr(self.stream, name)
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return self.stream.__exit__(*exc)
            def read(self, *args):
                os.truncate(spec, 0)
                return self.stream.read(*args)

        def opened(self, mode='r', *args, **kwargs):
            stream = real_open(self, mode, *args, **kwargs)
            return Truncated(stream) if same(self) and 'b' in mode else stream

        mmap.mmap, pathlib.Path.open = mapped, opened
        try:
            result = metadata.read_spec_points(images)
        except Exception as error:  # a typed outcome is fine; losing the process is not
            result = type(error).__name__
        print('survived', result)
    ''')
    environment = dict(os.environ)
    source = str(Path(metadata.__file__).resolve().parents[2])
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, (source, environment.get('PYTHONPATH'))))
    completed = subprocess.run([sys.executable, '-c', child, str(spec), *map(str, paths)],
                               capture_output=True, text=True, timeout=120, env=environment)
    assert (completed.returncode, completed.stdout.strip()) == (0, 'survived None'), completed.stderr[-2000:]
