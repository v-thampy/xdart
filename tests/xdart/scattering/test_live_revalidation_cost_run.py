"""LIVE-REVALIDATION-COST frozen live-Run oracle (Claude, October 15): R1-R7 and P1.

Real live Runs: a StandardRunExecutor watches a directory of real detector files (TIFF or raw with SPEC
rows or sidecars, a self-contained NeXus file, an Eiger master with external chunks). Work is counted by
delegating test-owned wrappers on the design's named seams; nothing on the seam is faked. A poll starts
when the Run's own thread asks the directory session for a refreshed observation. Its overhead is the
interval to the next poll, less the loop's own 0.1 s wait and the time the Run spends waiting for the new
frames' reduction (ReductionSession.drain); admission, construction, epoch commit and publication count.
Parent polarity at 8f4a48a3 is in the task's CHECKPOINT.md; P1's bars are frozen at the checkpoint.
"""
import os
import statistics
import sys
import threading
import time

import fabio
import h5py
import numpy as np
import pytest
import silx.io.specfile as silx_specfile

import xrd_tools.io.metadata as metadata
import xrd_tools.sources.directory_index as directory_index
import xrd_tools.sources.execution_graph as execution_graph
import xrd_tools.sources.image as image_source
import xrd_tools.sources.registry  # noqa: F401  # register built-in readers
from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_live_qualify_refusal_run import (
    SIDECARS, _labels, _said, _settle, _stop, _terminal, _terminals, _wait,
)
from tests.xdart.scattering.test_p1b_output_graph import _start
from xdart.gui.tabs.scattering import output_preflight
from xdart.gui.tabs.scattering.adapters import run_executor
from xdart.gui.tabs.scattering.adapters.dynamic_output import DynamicOutputAdapter
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus, ExecutorClosed
from xrd_tools.core.provenance import read_provenance
from xrd_tools.io.nexus import read_scan_metadata
from xrd_tools.reduction.core import ReductionSession
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xrd_tools.sources.base import BaseFrameSource
from xrd_tools.sources.directory_index import DirectoryIndex
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xrd_tools.sources.execution_graph import SourceFileState, graph_saturation, qualify_source_execution_graph
from xrd_tools.sources.nexus import NexusStackSource
from xrd_tools.sources.selection import DirectorySourceSpec, image_series_spec

POLL = run_executor._LIVE_DIRECTORY_POLL_S
CONTENT = ('qualify_read', 'reduce_read', 'spec_scan_read', 'spec_rows', 'silx_open', 'hdf5_open',
           'graph_validation')
PILATUS, TINY = (195, 487), (16, 16)
LABELS = ('del', 'i0', *(f'c{j}' for j in range(2, 22)))
SPEC_HEAD = '#F calibrant\n#E 1\n#O0 th  chi\n\n'


class _Work:
    """Delegating counters on the named owner seams, each call kept with its time."""

    def __init__(self, monkeypatch, *, counters=True):
        self.lock, self.calls, self.live_thread, self.entered = threading.Lock(), [], None, {}

        def record(seam, detail=None):
            with self.lock:
                self.calls.append((time.monotonic(), seam, detail))
            event = self.entered.get(seam)
            if event is not None:
                event.set()
        self.record = record

        def wrap(owner, name, seam, detail=None):
            real = getattr(owner, name)

            def counted(*args, **kwargs):
                record(seam, None if detail is None else detail(args))
                return real(*args, **kwargs)
            monkeypatch.setattr(owner, name, counted)

        real_live, real_observe = StandardRunExecutor._execute_live_directory, DirectoryIndexSession.observe
        real_drain, real_construct = ReductionSession.drain, StandardRunExecutor._construct

        def live(executor, run, receipt, deferred):
            self.live_thread = threading.get_ident()
            return real_live(executor, run, receipt, deferred)

        def observe(session, *, refresh=True):
            if refresh and threading.get_ident() == self.live_thread:
                record('poll')
            return real_observe(session, refresh=refresh)

        def drain(session, *args, **kwargs):
            started = time.monotonic()
            record('drain_start')
            try:
                return real_drain(session, *args, **kwargs)
            finally:
                record('drain', time.monotonic() - started)

        def construct(executor, run, **kwargs):
            record('construct')
            return real_construct(executor, run, **kwargs)

        def timed(seam, real):
            def call(*args, **kwargs):
                started = time.monotonic()
                try:
                    return real(*args, **kwargs)
                finally:
                    record(seam, time.monotonic() - started)
            return call
        monkeypatch.setattr(StandardRunExecutor, '_execute_live_directory', live)
        monkeypatch.setattr(DirectoryIndexSession, 'observe', observe)
        monkeypatch.setattr(ReductionSession, 'drain', drain)
        monkeypatch.setattr(StandardRunExecutor, '_construct', construct)
        monkeypatch.setattr(DynamicOutputAdapter, 'commit_epoch',
                            timed('commit_epoch', DynamicOutputAdapter.commit_epoch))
        monkeypatch.setattr(DynamicOutputAdapter, 'finish_all', timed('finish', DynamicOutputAdapter.finish_all))
        monkeypatch.setattr(DirectoryIndex, 'poll', timed('discovery', DirectoryIndex.poll))
        real_admitted = StandardRunExecutor._execute_admitted

        def admitted(executor, run):
            try:
                return real_admitted(executor, run)
            finally:
                record('work_exit')  # Stop's t1: the worker has left its work
        monkeypatch.setattr(StandardRunExecutor, '_execute_admitted', admitted)
        monkeypatch.setattr(StandardRunExecutor, '_cleanup', timed('cleanup', StandardRunExecutor._cleanup))
        if not counters:
            return
        wrap(metadata, 'read_image_metadata_observed', 'qualify_read', lambda a: str(a[0]))
        wrap(image_source, 'read_image_metadata_observed', 'reduce_read', lambda a: str(a[0]))
        wrap(metadata, '_read_spec_scan', 'spec_scan_read', lambda a: _revision(a[0]))
        wrap(metadata, '_spec_rows_values', 'spec_rows', lambda a: len(a[2]))
        wrap(silx_specfile, 'SpecFile', 'silx_open')
        wrap(execution_graph, '_HDF5_FILE_OPEN', 'hdf5_open', lambda a: str(a[0]))
        wrap(execution_graph, 'validate_source_execution_graph', 'graph_validation')
        wrap(output_preflight, 'validate_source_execution_graph', 'graph_validation')
        real_owner, real_capture, real_to_scan = (DirectoryIndexSession._observe_on_owner,
                                                  SourceFileState.capture, BaseFrameSource.to_scan)

        def observe_owner(session, generation, config, refresh):
            record('observation', refresh)
            return real_owner(session, generation, config, refresh)

        def capture(cls, path):
            record('capture')
            return real_capture(path)

        def to_scan(source, *args, **kwargs):
            scan = real_to_scan(source, *args, **kwargs)
            record('to_scan_frames', len(scan.frames))
            return scan
        monkeypatch.setattr(DirectoryIndexSession, '_observe_on_owner', observe_owner)
        monkeypatch.setattr(SourceFileState, 'capture', classmethod(capture))
        monkeypatch.setattr(BaseFrameSource, 'to_scan', to_scan)

    def times(self, seam, start=0., end=float('inf')):
        with self.lock:
            return [(at, detail) for at, name, detail in self.calls if name == seam and start <= at < end]

    def polls(self, after=0.):
        return [at for at, _detail in self.times('poll', after)]

    def window(self, start, end):
        """Every counted call in [start, end), by seam."""
        found = {}
        with self.lock:
            for at, name, detail in self.calls:
                if start <= at < end:
                    found.setdefault(name, []).append(detail)
        return found

    def await_polls(self, after, count, deadline):
        while len(self.polls(after)) < count:
            assert time.monotonic() < deadline, f'{len(self.polls(after))} of {count} polls before the deadline'
            time.sleep(.01)
        return self.polls(after)[:count]

    def overhead(self, start, end):
        """The overhead of the polls starting in [start, end): the interval less each loop wait and the new
        frames' reduction."""
        waits = len([at for at in self.polls(start) if at < end])
        return end - start - POLL * waits - sum(detail for _at, detail in self.times('drain', start, end))

    def net(self, seam, start, end):
        """Each ``seam`` call ending in [start, end): its duration less the reduction drained inside it."""
        return [took - sum(drained for _at, drained in self.times('drain', done - took, done + 1e-9))
                for done, took in self.times(seam, start, end)]


def _revision(path):
    state = os.stat(path)
    return state.st_size, state.st_mtime_ns, state.st_ino


def _poni(tmp_path, shape):
    path = tmp_path / 'cal.poni'
    if shape == PILATUS:
        write_poni(path)
    else:
        path.write_text(
            'poni_version: 2\nDetector: Detector\nDetector_config: {"pixel1": 0.000172, "pixel2": 0.000172, '
            f'"max_shape": [{shape[0]}, {shape[1]}]}}\nDistance: 0.1234\nPoni1: 0.0014\nPoni2: 0.0014\n'
            'Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1.0e-10\n')
    return path


def _tiff(path, value, shape):
    partial = path.with_suffix('.part')
    fabio.tifimage.TifImage(data=np.full(shape, value, dtype=np.uint16)).write(str(partial))
    partial.replace(path)


def _raw(path, value, shape):
    partial = path.with_suffix('.part')
    np.full(shape, value, dtype=np.int32).tofile(partial)
    partial.replace(path)


class _Spec:
    """SPEC + TIFF or raw images, SSRL names, 22 columns; i0 of point k of scan s is 100 * s + k. With ``energy``
    the last column is ``energy``: k + 21 keV, so every frame has its own."""

    def __init__(self, raw, shape, suffix='.tif', history_bytes=0, energy=False):
        self.raw, self.shape, self.suffix, self.path = raw, shape, suffix, raw / 'calibrant'
        self.labels = (*LABELS[:-1], 'energy') if energy else LABELS
        parts, size, scan = [SPEC_HEAD], len(SPEC_HEAD), 1000
        while size < history_bytes:  # earlier scans the Live Run does not reduce: no image names them
            scan += 1
            text = self._header(scan) + ''.join(self._line(scan, k) for k in range(400))
            parts.append(text)
            size += len(text)
        self.path.write_text(''.join(parts))

    def _header(self, scan):
        return f'#S {scan} ascan th 0 1 10 1\n#P0 0.5 1\n#N {len(LABELS)}\n#L ' + '  '.join(self.labels) + '\n'

    @staticmethod
    def _line(scan, k):
        return ' '.join([f'{k / 10:.1f}', str(100 * scan + k), *(str(k + j) for j in range(2, 22))]) + '\n'

    def scan(self, scan):
        with self.path.open('a') as stream:
            stream.write(self._header(scan))

    def row(self, scan, k):
        with self.path.open('a') as stream:
            stream.write(self._line(scan, k))

    def image(self, scan, k):
        target = self.raw / f'calibrant_scan{scan:03d}_{k:04d}{self.suffix}'
        (_tiff if self.suffix == '.tif' else _raw)(target, k + 1, self.shape)
        return target

    def frame(self, scan, k):
        self.row(scan, k)
        return self.image(scan, k)


class _Sidecar:
    """TIFF images with .txt sidecars (I0 of frame k is k)."""

    def __init__(self, raw, shape, stem='scan'):
        self.raw, self.shape, self.stem = raw, shape, stem

    def sidecar(self, k, i0=None):
        (self.raw / f'{self.stem}_{k:04d}.txt').write_text(SIDECARS['txt'](float(k if i0 is None else i0)))

    def image(self, k):
        target = self.raw / f'{self.stem}_{k:04d}.tif'
        _tiff(target, k, self.shape)
        return target

    def frame(self, k):
        self.sidecar(k)
        return self.image(k)


def _open_for_write(path, mode):
    deadline = time.monotonic() + 5.
    while True:
        try:
            return h5py.File(path, mode)
        except OSError:  # a Live read holds the file for a moment
            assert time.monotonic() < deadline, f'{path} stayed locked'
            time.sleep(.01)


class _Nexus:
    """One self-contained NeXus file whose detector stack grows in place."""

    def __init__(self, raw, shape, frames):
        self.path, self.shape = raw / 'scan_0.nxs', shape
        with h5py.File(self.path, 'w') as handle:
            detector = handle.create_group('entry').create_group('instrument').create_group('detector')
            detector.create_dataset('data', data=np.ones((frames, *shape), dtype=np.uint16),
                                    maxshape=(None, *shape), chunks=(1, *shape))
            positioners = handle['entry/instrument'].create_group('positioners')
            positioners.create_group('halpha').create_dataset('value', data=0.2)

    def grow(self, count=1):
        with _open_for_write(self.path, 'a') as handle:
            data = handle['entry/instrument/detector/data']
            data.resize((data.shape[0] + count, *self.shape))
            data[-count:] = 2


class _Eiger:
    """An Eiger master with external data chunks; the last chunk grows."""

    def __init__(self, raw, shape, frames, chunk=100, stem='scan'):
        self.raw, self.shape, self.stem, self.chunks = raw, shape, stem, []
        self.master = raw / f'{stem}_master.h5'
        remaining = frames
        while remaining > 0:
            self._chunk(min(chunk, remaining))
            remaining -= min(chunk, remaining)
        with h5py.File(self.master, 'w') as handle:
            entry = handle.create_group('entry')
            entry.attrs['NX_class'] = 'NXentry'
            data = entry.create_group('data')
            data.attrs['NX_class'] = 'NXdata'
            for index, path in enumerate(self.chunks, 1):
                data[f'data_{index:06d}'] = h5py.ExternalLink(path.name, '/entry/data/data')

    def _chunk(self, frames):
        path = self.raw / f'{self.stem}_data_{len(self.chunks) + 1:06d}.h5'
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('entry/data/data', data=np.ones((frames, *self.shape), dtype=np.uint16),
                                  maxshape=(None, *self.shape), chunks=(1, *self.shape))
        self.chunks.append(path)

    def grow(self, count=1):
        with _open_for_write(self.chunks[-1], 'a') as handle:
            data = handle['entry/data/data']
            data.resize((data.shape[0] + count, *self.shape))
            data[-count:] = 2


def _intent(tmp_path, raw, shape, *, suffix='.tif', fmt='spec', mode='Overwrite', gi=None, source=None,
            processing='Int 1D', one=None, two=None):
    return RunIntent(
        source_spec=source or DirectorySourceSpec(raw, suffixes=(suffix,), metadata_format=fmt),
        poni_file=str(_poni(tmp_path, shape)), project_root=str(raw), save_path=str(tmp_path / 'processed'),
        output_mode=mode, processing_mode=processing, live_mode=True, max_cores=1,
        bai_1d_args=one or {'npt': 8, 'method': 'numpy'}, **({} if gi is None else {'gi': gi}),
        **({} if two is None else {'bai_2d_args': two}))


def _counted(tmp_path, stem):
    """(labels, per-frame record) of one output, read after the Run closed."""
    path = tmp_path / 'processed' / f'{stem}_int1d.nexus'
    if not path.exists():
        return None, None
    with h5py.File(path, 'r') as handle:
        labels = tuple(int(value) for value in handle['entry/integrated_1d/frame_index'][()])
    return labels, read_scan_metadata(path)


def _column(record, key):
    return None if record is None or key not in record else tuple(float(value) for value in record[key].values)


def _settled(executor, identity, frames, deadline):
    """The published attempts once none is partial and they hold ``frames`` frames in all."""
    while True:
        attempts = executor.processed_live_revisions(identity)
        if attempts and not any(value.partial for value in attempts) and sum(
                value.decision.item.source_stamp.frame_count for value in attempts) >= frames:
            return attempts
        assert time.monotonic() < deadline, 'the published attempts never settled'
        time.sleep(.01)


def _differential(attempts):
    """Every settled graph equals a fresh full qualification, in-memory facts included: a TIFF series' members, and
    a container at its logical frame count (each frame's index, SPEC row and extent, and the reader policy)."""
    problems = []
    for attempt in attempts:
        graph = attempt.decision.item.graph
        frames = graph.stamp.frame_count
        fresh = qualify_source_execution_graph(
            graph.execution_source, infer_point_images=graph.infers_point_images,
            published_members=len(graph.stamp.members) or frames)
        mine, theirs = graph.stamp.metadata_sources, fresh.stamp.metadata_sources
        if graph.stamp.members or fresh.stamp.frame_count == frames:
            if fresh != graph:
                problems.append(f'{graph.group_key}: graph differs from a fresh qualification')
        elif fresh.stamp.frame_count < frames or (fresh.execution_source, fresh.reader_binding, fresh.group_key) != (
                graph.execution_source, graph.reader_binding, graph.group_key):  # a container that grew since
            problems.append(f'{graph.group_key}: the container no longer extends its settled graph')
        if fresh.infers_point_images != graph.infers_point_images:
            problems.append(f'{graph.group_key}: infers_point_images differs')
        if not graph.stamp.members and (len(mine) not in (0, frames) or len(theirs) < len(mine)):
            problems.append(f'{graph.group_key}: {len(mine)} admitted rows for {frames} frames')
        for ours, fresh_row in zip(mine, theirs):
            extent = fresh_row.spec_extent
            if not graph.stamp.members and extent is not None and ours.spec_extent is not None:
                extent = extent.through(len(ours.spec_extent.marks) - 1)
            if (ours.frame_index, ours.spec_extent, dict(ours.spec_values or {})) != (
                    fresh_row.frame_index, extent, dict(fresh_row.spec_values or {})):
                problems.append(f'{ours.source_path} frame {ours.frame_index}: retained SPEC row/extent differ')
    return problems


def _content(found, published=()):
    """Counted content work, and the published members whose metadata was read again."""
    work = {seam: len(found[seam]) for seam in CONTENT if found.get(seam)}
    reread = sorted({detail for seam in ('qualify_read', 'reduce_read') for detail in found.get(seam, ())
                     if detail in published})
    return work, reread


def _report(row, phase, found):
    """Print one phase's counted work (calls per seam; rows, frames and seconds summed) for the record."""
    summed = {'spec_rows', 'to_scan_frames', 'drain', 'commit_epoch', 'finish', 'discovery', 'cleanup'}
    counts = {seam: round(sum(values), 3) if seam in summed else len(values)
              for seam, values in sorted(found.items()) if seam != 'drain_start'}
    print(f'WORK {row} {phase}: {counts}')


def _images(raw):
    return {str(path) for path in raw.iterdir() if path.suffix in ('.tif', '.raw')}


def _count_labels(stem, count):
    return lambda values: len(set(_labels(values, stem))) >= count or _terminal(values)


# --- R1: one lifecycle per source family (idle, pending, 1 and k arrivals) --------------------------------------

FAMILIES = ('spec_tiff', 'spec_raw', 'txt', 'auto', 'nexus', 'eiger')


def _family(name, tmp_path, raw, frames):
    """(intent, output stem, first label, arrive(k), pending(k) or None, land(k), expected i0 or None)."""
    if name in ('spec_tiff', 'spec_raw'):
        spec = _Spec(raw, PILATUS, suffix='.tif' if name == 'spec_tiff' else '.raw')
        spec.scan(1)
        for k in range(frames):
            spec.frame(1, k)
        return (_intent(tmp_path, raw, PILATUS, suffix=spec.suffix), 'calibrant_scan001', 1,
                lambda k: spec.frame(1, k), lambda k: spec.image(1, k), lambda k: spec.row(1, k),
                lambda count: tuple(float(100 + k) for k in range(count)))
    if name in ('txt', 'auto'):
        cars = _Sidecar(raw, PILATUS)
        for k in range(frames):
            cars.frame(k)
        return (_intent(tmp_path, raw, PILATUS, fmt=name), 'scan', 1, cars.frame,
                cars.image if name == 'txt' else None, cars.sidecar,
                lambda count: tuple(float(k) for k in range(count)))
    if name == 'nexus':
        stack = _Nexus(raw, PILATUS, frames)
        return (_intent(tmp_path, raw, PILATUS, suffix='.nxs', fmt=None), 'scan_0', 0,
                lambda _k: stack.grow(), None, None, None)
    chunks = _Eiger(raw, PILATUS, frames, chunk=max(1, frames // 2))
    return (_intent(tmp_path, raw, PILATUS, suffix='_master.h5', fmt=None), 'scan', 0,
            lambda _k: chunks.grow(), None, None, None)


@pytest.mark.parametrize('name', FAMILIES)
def test_r1_a_live_family_does_only_new_work_at_idle_pending_and_arrival(tmp_path, monkeypatch, name):
    raw = tmp_path / 'raw'
    raw.mkdir()
    n, k = 6, 3
    intent, stem, first, arrive, pending, land, expected = _family(name, tmp_path, raw, n)
    work = _Work(monkeypatch)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, intent, request_value=7100 + FAMILIES.index(name))
    events, problems, deadline = [], [], time.monotonic() + 180.
    try:
        _wait(executor, events, _count_labels(stem, n), 60.)
        _settled(executor, identity, n, deadline)
        published = _images(raw)
        polls = work.await_polls(time.monotonic(), 12, deadline)
        found = work.window(polls[1], polls[11])
        _report(name, 'idle_10_polls', found)
        idle, _ = _content(found)
        if idle:
            problems.append(f'idle: ten unchanged polls did content work {idle}')
        if pending is not None:
            pending(n)
            polls = work.await_polls(time.monotonic(), 12, deadline)
            found = work.window(polls[2], polls[11])
            _report(name, 'pending_9_polls', found)
            waited, reread = _content(found, published)
            if waited or reread:
                problems.append(f'pending: nine unchanged waits did content work {waited}, re-read {len(reread)}')
            if len(found.get('observation', ())) > 2 * 9:
                problems.append(f'pending: nine waits rebuilt the observation {len(found["observation"])} times')
            mark = time.monotonic()
            land(n)
        else:
            mark = time.monotonic()
            arrive(n)
        _wait(executor, events, _count_labels(stem, n + 1), 60.)
        attempts = _settled(executor, identity, n + 1, deadline)
        found = work.window(mark, time.monotonic())
        _report(name, 'arrival_1', found)
        constructs = [at for at, _detail in work.times('construct', mark)]
        if not any(mark <= at - took and at <= constructs[0] for at, took in work.times('discovery', mark)):
            problems.append('the landed frame was admitted without a real listing after it landed')
        _arrived, reread = _content(found, published)
        if reread:
            problems.append(f'one arrival re-read the published metadata of {len(reread)} members')
        if sum(found.get('spec_rows', ())) > 1:
            problems.append(f'one arrival extracted {sum(found["spec_rows"])} SPEC rows')
        revisions = found.get('spec_scan_read', [])
        if len(revisions) > len(set(revisions)):
            problems.append(f'one arrival read {len(set(revisions))} SPEC revisions {len(revisions)} times')
        if len(found.get('hdf5_open', ())) > 2:
            problems.append(f'one container arrival opened HDF5 {len(found["hdf5_open"])} times')
        problems += _differential(attempts)
        published = _images(raw)
        mark = time.monotonic()
        for index in range(n + 1, n + 1 + k):
            arrive(index)
        total = n + 1 + k
        _wait(executor, events, _count_labels(stem, total), 60.)
        attempts = _settled(executor, identity, total, deadline)
        found = work.window(mark, time.monotonic())
        _report(name, f'arrival_{k}', found)
        _arrived, reread = _content(found, published)
        if reread:
            problems.append(f'{k} arrivals re-read the published metadata of {len(reread)} members')
        if sum(found.get('spec_rows', ())) > k:
            problems.append(f'{k} arrivals extracted {sum(found["spec_rows"])} SPEC rows')
        problems += _differential(attempts)
    finally:
        terminal = _stop(executor, identity, events)
    print(f'PROBLEMS {name}: {problems}')
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    labels, record = _counted(tmp_path, stem)
    assert labels == tuple(range(first, first + n + 1 + k)), labels
    if expected is not None:
        assert _column(record, 'i0' if name.startswith('spec') else 'I0') == expected(n + 1 + k)
    assert not problems, problems


# --- R2: a later SPEC scan, and the cross-group drift sweep ------------------------------------------------------

def test_r2a_a_later_spec_scan_gets_its_own_output_without_rereading_the_settled_scan(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    spec = _Spec(raw, PILATUS)
    spec.scan(1)
    for k in range(5):
        spec.frame(1, k)
    work = _Work(monkeypatch)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, PILATUS), request_value=7110)
    events, deadline = [], time.monotonic() + 120.
    try:
        _wait(executor, events, _count_labels('calibrant_scan001', 5), 60.)
        _settled(executor, identity, 5, deadline)
        scan1, mark = _images(raw), time.monotonic()
        spec.scan(2)
        for k in range(4):
            spec.frame(2, k)
        _wait(executor, events, _count_labels('calibrant_scan002', 4), 60.)
        _settled(executor, identity, 9, deadline)
        polls = work.await_polls(time.monotonic(), 6, deadline)
        found = work.window(mark, polls[-1])
        _report('r2a', 'scan2_arrival_and_5_polls', found)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _column(_counted(tmp_path, 'calibrant_scan001')[1], 'i0') == tuple(100. + k for k in range(5))
    assert _column(_counted(tmp_path, 'calibrant_scan002')[1], 'i0') == tuple(200. + k for k in range(4))
    reread = sorted({detail for seam in ('qualify_read', 'reduce_read') for detail in found.get(seam, ())
                     if detail in scan1})
    rows = sum(found.get('spec_rows', ()))
    assert (len(reread), rows <= 4) == (0, True), (len(reread), rows)


def test_r2b_every_settled_output_is_swept_for_drift_within_one_second(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    for scan in (1, 2, 3):
        cars = _Sidecar(raw, PILATUS, stem=f'scan{scan}')
        for k in range(3):
            cars.sidecar(k, 10 * scan + k)
            cars.image(k)
    work = _Work(monkeypatch)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, PILATUS, fmt='txt'), request_value=7111)
    events, deadline = [], time.monotonic() + 120.
    try:
        _wait(executor, events, lambda values: all(len(set(_labels(values, f'scan{s}'))) >= 3
                                                   for s in (1, 2, 3)) or _terminal(values), 60.)
        _settled(executor, identity, 9, deadline)
        polls = work.await_polls(time.monotonic(), 14, deadline)
        found = work.window(polls[2], polls[13])
        _report('r2b', 'idle_11_polls_3_outputs', found)
        idle, _ = _content(found)
        # A new frame whose directory stamp lags (held at its pre-write identity) is still listed within 1 s.
        real_identity = getattr(directory_index, 'directory_identity', None)
        held = {os.fspath(raw): real_identity(raw)} if real_identity else {}

        def lagging(path):
            work.record('identity_held' if os.fspath(path) in held else 'identity')
            return held.get(os.fspath(path)) or real_identity(path)
        monkeypatch.setattr(directory_index, 'directory_identity', lagging, raising=False)
        written = time.monotonic()
        _Sidecar(raw, PILATUS, stem='scan2').frame(3)
        _wait(executor, events, lambda values: len(set(_labels(values, 'scan2'))) >= 4 or _terminal(values), 10.)
        held.clear()
        listed = [at for at, took in work.times('discovery', written) if at - took >= written]
        lag = min(listed) - written if listed else None
        (raw / 'scan1_0001.txt').write_text(SIDECARS['txt'](99.))  # a published sidecar changes
        changed = time.monotonic()
        _wait(executor, events, lambda values: _said(values, 'stopped growing') or _terminal(values), 10.)
        noticed = time.monotonic() - changed
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _said(events, 'scan1_0001.txt'), 'the drift names the changed sidecar'
    assert (noticed <= 1. + 2 * POLL, idle) == (True, {}), (round(noticed, 3), idle)
    assert lag is not None and lag <= 1., f'a lagging directory stamp hid the new frame for {lag} s'
    assert work.times('identity_held'), 'the held directory identity was never consulted'


# --- R3: invalidation after reuse is established (the parent's exact outcomes preserved) ------------------------

def _settle_reuse(executor, identity, work, events, frames, deadline):
    _settled(executor, identity, frames, deadline)
    work.await_polls(time.monotonic(), 14, deadline)  # one full drift sweep and ten idle polls
    _settle(executor, events, .2)


def _sidecar_run(tmp_path, monkeypatch, request_value, frames=3):
    raw = tmp_path / 'raw'
    raw.mkdir()
    cars = _Sidecar(raw, PILATUS)
    for k in range(frames):
        cars.frame(k)
    work = _Work(monkeypatch, counters=False)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, PILATUS, fmt='txt'), request_value=request_value)
    events = []
    _wait(executor, events, _count_labels('scan', frames), 60.)
    _settle_reuse(executor, identity, work, events, frames, time.monotonic() + 60.)
    return raw, cars, executor, identity, events


def _same_size_rewrite(path, old, new):
    state = path.stat()
    text = path.read_text()
    assert text.count(old) == 1 and len(old) == len(new)
    path.write_text(text.replace(old, new))
    os.utime(path, ns=(state.st_atime_ns, state.st_mtime_ns))  # restored mtime: only ctime records the change
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (state.st_size, state.st_mtime_ns)


def _blocked_then_skips(executor, identity, events, later_frame):
    """Wait for the output to stop growing, offer a later frame, and Stop."""
    _wait(executor, events, lambda values: _said(values, 'stopped growing') or _terminal(values), 10.)
    later_frame()
    _settle(executor, events, .8)
    return _stop(executor, identity, events)


@pytest.mark.parametrize('case', ['same_size_restored_mtime', 'removed'])
def test_r3a_a_published_sidecar_change_stops_its_output(tmp_path, monkeypatch, case):
    raw, cars, executor, identity, events = _sidecar_run(tmp_path, monkeypatch, 7120 + (case == 'removed'))
    try:
        if case == 'removed':
            (raw / 'scan_0001.txt').unlink()
        else:
            _same_size_rewrite(raw / 'scan_0001.txt', 'I0 = 1.0', 'I0 = 7.0')
        terminal = _blocked_then_skips(executor, identity, events, lambda: cars.frame(3))
    finally:
        if not _terminal(events):
            _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _said(events, 'scan_0001.txt') and _counted(tmp_path, 'scan')[0] == (1, 2, 3)


@pytest.mark.parametrize('case', ['row', 'file_header'])
def test_r3b_a_published_spec_row_or_file_header_rewrite_stops_its_output(tmp_path, monkeypatch, case):
    raw = tmp_path / 'raw'
    raw.mkdir()
    spec = _Spec(raw, PILATUS)
    spec.scan(1)
    for k in range(3):
        spec.frame(1, k)
    work = _Work(monkeypatch, counters=False)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, PILATUS), request_value=7122 + (case == 'file_header'))
    events = []
    try:
        _wait(executor, events, _count_labels('calibrant_scan001', 3), 60.)
        _settle_reuse(executor, identity, work, events, 3, time.monotonic() + 60.)
        old, new = ('0.1 101 ', '0.1 109 ') if case == 'row' else ('#O0 th  chi', '#O0 ta  chi')
        _same_size_rewrite(spec.path, old, new)
        terminal = _blocked_then_skips(executor, identity, events, lambda: spec.frame(1, 3))
    finally:
        if not _terminal(events):
            _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _said(events, 'stopped growing') and _counted(tmp_path, 'calibrant_scan001')[0] == (1, 2, 3)


#: Today's exact outcomes, frozen from the parent run at 8f4a48a3 (draft run dev02; see CHECKPOINT.md).
R3C_PARENT = (['STOPPED'], (1, 2, 3, 4), (0.0, 1.0, 2.0, 3.0), False)
R3D_PARENT = {
    'raw_rewrite': (['FAILED'], (1, 2, 3), False, ['same-extent source identity changed']),
    'leaf_symlink_to_hardlink': (['FAILED'], (1, 2, 3), False, ['same-extent source identity changed']),
    'regular_to_symlink': (['FAILED'], (1, 2, 3), False, ['source does not extend exact image-member history']),
}
R3E_PARENT = (['FAILED'], (0, 1, 2, 3), ['source does not extend exact external-member history'])


def test_r3c_under_auto_a_competing_sidecar_keeps_todays_selection(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    cars = _Sidecar(raw, PILATUS)
    for k in range(3):  # the replace form ("scan_0000.txt") latches Auto's convention for this folder
        cars.frame(k)
    work = _Work(monkeypatch, counters=False)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, PILATUS, fmt='auto'), request_value=7124)
    events = []
    try:
        _wait(executor, events, _count_labels('scan', 3), 60.)
        _settle_reuse(executor, identity, work, events, 3, time.monotonic() + 60.)
        (raw / 'scan_0001.tif.txt').write_text(SIDECARS['txt'](50.))  # a competing append-form candidate
        (raw / 'scan_0003.tif.txt').write_text(SIDECARS['txt'](53.))
        cars.frame(3)
        _wait(executor, events, lambda values: len(set(_labels(values, 'scan'))) >= 4 or _terminal(values)
              or _said(values, 'stopped growing'), 20.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    labels, record = _counted(tmp_path, 'scan')
    outcome = ([kind for kind, _detail in terminal], labels, _column(record, 'I0'), _said(events, 'stopped growing'))
    assert outcome == R3C_PARENT, outcome


@pytest.mark.parametrize('case', ['raw_rewrite', 'leaf_symlink_to_hardlink', 'regular_to_symlink'])
def test_r3d_a_published_raw_member_change_keeps_its_exact_outcome(tmp_path, monkeypatch, case):
    raw, store = tmp_path / 'raw', tmp_path / 'store'
    raw.mkdir()
    store.mkdir()
    cars = _Sidecar(raw, PILATUS)
    for k in range(3):
        cars.sidecar(k)
        if case == 'leaf_symlink_to_hardlink':
            _tiff(store / f'img_{k}.tif', k, PILATUS)
            (raw / f'scan_{k:04d}.tif').symlink_to(store / f'img_{k}.tif')
        else:
            cars.image(k)
    work = _Work(monkeypatch, counters=False)
    executor = StandardRunExecutor(join_timeout=2.)
    cases = ('raw_rewrite', 'leaf_symlink_to_hardlink', 'regular_to_symlink')
    identity = _start(executor, _intent(tmp_path, raw, PILATUS, fmt='txt'), request_value=7125 + cases.index(case))
    events = []
    try:
        _wait(executor, events, _count_labels('scan', 3), 60.)
        _settle_reuse(executor, identity, work, events, 3, time.monotonic() + 60.)
        member = raw / 'scan_0001.tif'
        if case == 'raw_rewrite':
            state = member.stat()
            fabio.tifimage.TifImage(data=np.full(PILATUS, 77, dtype=np.uint16)).write(str(member))
            os.utime(member, ns=(state.st_atime_ns, state.st_mtime_ns))
        elif case == 'leaf_symlink_to_hardlink':
            os.link(store / 'img_1.tif', store / 'img_1_again.tif')  # same inode and bytes, another path
            member.unlink()
            member.symlink_to(store / 'img_1_again.tif')
        else:
            copy = store / 'copy_1.tif'
            copy.write_bytes(member.read_bytes())
            member.unlink()
            member.symlink_to(copy)
        _settle(executor, events, 1.5)
        cars.frame(3)
        _wait(executor, events, lambda values: len(set(_labels(values, 'scan'))) >= 4 or _terminal(values)
              or _said(values, 'stopped growing'), 20.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    # The same-extent refusal now names what changed (1.2.4); the frozen outcome keeps its stem.
    stem = 'same-extent source identity changed: '
    named = [detail[len(stem):] for _kind, detail in terminal if detail.startswith(stem)]
    assert all(change.strip() for change in named), terminal
    # ... and the image-member refusal names the first image that differs (2026-10-23).
    member_stem = 'source does not extend exact image-member history: '
    outcome = ([kind for kind, _detail in terminal], _counted(tmp_path, 'scan')[0], _said(events, 'stopped growing'),
               [stem[:-2] if detail.startswith(stem) else member_stem[:-2] if detail.startswith(member_stem)
                else detail for _kind, detail in terminal])
    assert outcome == R3D_PARENT[case], outcome


@pytest.mark.parametrize('case', ['auto_rejected_becomes_plausible', 'same_id_adapter_replacement'])
def test_r3f_a_discovery_or_registry_change_is_proved_strictly_again(tmp_path, monkeypatch, case):
    """No outcome changes today, but the next proof must not reuse: the member is read strictly again."""
    from dataclasses import replace

    from xrd_tools.sources import adapters as source_adapters
    raw = tmp_path / 'raw'
    raw.mkdir()
    cars = _Sidecar(raw, PILATUS)
    if case == 'auto_rejected_becomes_plausible':  # an append-form candidate Auto rejects (one pair)
        (raw / 'scan_0001.tif.txt').write_text('note = 1\n')
    for k in range(3):
        cars.frame(k)
    work = _Work(monkeypatch)
    executor = StandardRunExecutor(join_timeout=2.)
    fmt = 'auto' if case == 'auto_rejected_becomes_plausible' else 'txt'
    identity = _start(executor, _intent(tmp_path, raw, PILATUS, fmt=fmt), request_value=7129 + (fmt == 'txt') * 10)
    events = []
    try:
        _wait(executor, events, _count_labels('scan', 3), 60.)
        _settle_reuse(executor, identity, work, events, 3, time.monotonic() + 60.)
        changed = time.monotonic()
        if case == 'auto_rejected_becomes_plausible':
            (raw / 'scan_0001.tif.txt').write_text(SIDECARS['txt'](41.))
        else:
            owner = source_adapters.candidate_owner(raw / 'scan_0000.tif')
            entry = source_adapters._ADAPTERS[owner.id]
            monkeypatch.setitem(source_adapters._ADAPTERS, owner.id, source_adapters._Entry(
                replace(entry.adapter), entry.builtin, entry.seq + 100000))
        _settle(executor, events, 1.5)
        member = str(raw / 'scan_0001.tif')
        reproved = [detail for _at, detail in work.times('qualify_read', changed) if detail == member]
        cars.frame(3)
        _wait(executor, events, lambda values: len(set(_labels(values, 'scan'))) >= 4 or _terminal(values)
              or _said(values, 'stopped growing'), 20.)
        _settle(executor, events, .3)
    finally:
        terminal = _stop(executor, identity, events)
    labels, record = _counted(tmp_path, 'scan')
    outcome = ([kind for kind, _detail in terminal], labels, _column(record, 'I0'), _said(events, 'stopped growing'))
    assert (outcome, bool(reproved)) == ((['STOPPED'], (1, 2, 3, 4), (0., 1., 2., 3.), False), True), (
        outcome, len(reproved))


def test_r3e_a_rewritten_published_eiger_chunk_keeps_its_exact_outcome(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    chunks = _Eiger(raw, PILATUS, 4, chunk=2)
    work = _Work(monkeypatch, counters=False)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, PILATUS, suffix='_master.h5', fmt=None),
                      request_value=7128)
    events = []
    try:
        _wait(executor, events, _count_labels('scan', 4), 60.)
        _settle_reuse(executor, identity, work, events, 4, time.monotonic() + 60.)
        with _open_for_write(chunks.chunks[0], 'a') as handle:
            handle['entry/data/data'][1] = 9  # a published frame's bytes change
        _settle(executor, events, 1.5)
        chunks.grow()
        _wait(executor, events, lambda values: len(set(_labels(values, 'scan'))) >= 5 or _terminal(values)
              or _said(values, 'stopped growing'), 20.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    outcome = ([kind for kind, _detail in terminal], _counted(tmp_path, 'scan')[0],
               [detail for _kind, detail in terminal])
    assert outcome == R3E_PARENT, outcome


# --- R4: races and refusals --------------------------------------------------------------------------------------

def test_r4a_a_published_sidecar_churning_during_idle_proofs_ends_fenced(tmp_path, monkeypatch):
    raw, _cars, executor, identity, events = _sidecar_run(tmp_path, monkeypatch, 7130)
    stop, path = threading.Event(), raw / 'scan_0002.txt'

    def churn():
        value = 0
        while not stop.is_set():
            value += 1
            path.write_text(SIDECARS['txt'](float(value % 10)))
            time.sleep(.013)
    thread = threading.Thread(target=churn)
    try:
        thread.start()
        _settle(executor, events, 1.2)
        stop.set()
        thread.join()
        _wait(executor, events, lambda values: _said(values, 'stopped growing') or _terminal(values), 10.)
    finally:
        stop.set()
        thread.join()
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _said(events, 'scan_0002.txt') and _counted(tmp_path, 'scan')[0] == (1, 2, 3)


def test_r4b_spec_rows_appended_while_frames_wait_never_fence_them(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    spec = _Spec(raw, PILATUS)
    spec.scan(1)
    for k in range(3):
        spec.frame(1, k)
    work = _Work(monkeypatch, counters=False)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, PILATUS), request_value=7131)
    events = []
    try:
        _wait(executor, events, _count_labels('calibrant_scan001', 3), 60.)
        _settle_reuse(executor, identity, work, events, 3, time.monotonic() + 60.)
        for k in range(3, 9):  # images land first; their rows trickle in while the Run proves its prefix
            spec.image(1, k)
        for k in range(3, 9):
            time.sleep(.07)
            spec.row(1, k)
        _wait(executor, events, lambda values: len(set(_labels(values, 'calibrant_scan001'))) >= 9
              or _terminal(values) or _said(values, 'stopped growing'), 30.)
        _settle(executor, events, .3)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert not _said(events, 'stopped growing')
    assert _column(_counted(tmp_path, 'calibrant_scan001')[1], 'i0') == tuple(100. + k for k in range(9))


def test_r4c_a_complete_malformed_spec_row_for_a_new_frame_fails_loudly(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    spec = _Spec(raw, PILATUS)
    spec.scan(1)
    for k in range(3):
        spec.frame(1, k)
    work = _Work(monkeypatch, counters=False)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, PILATUS), request_value=7132)
    events = []
    try:
        _wait(executor, events, _count_labels('calibrant_scan001', 3), 60.)
        _settle_reuse(executor, identity, work, events, 3, time.monotonic() + 60.)
        with spec.path.open('a') as stream:
            stream.write('0.3 103 not-a-number\n')
        spec.image(1, 3)
        _wait(executor, events, _terminal, 30.)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal][-1:] == ['FAILED'], terminal
    assert 'row 4' in str(terminal) and _counted(tmp_path, 'calibrant_scan001')[0] == (1, 2, 3)


def test_r4d_a_new_spec_member_waits_for_a_second_real_listing(tmp_path, monkeypatch):
    """d2c853da: a member first seen by a poll's listing is admitted only once a later real listing (listing reuse
    on) has seen it too, so one listing that misses earlier members never publishes a gapped series; the saved
    labels, sources and i0 stay in creation order."""
    raw = tmp_path / 'raw'
    raw.mkdir()
    spec = _Spec(raw, PILATUS)
    spec.scan(1)
    for k in range(3):
        spec.frame(1, k)
    listed, admitted, poll, construct = [], [], DirectoryIndex.poll, StandardRunExecutor._construct

    def listing(index):
        snapshot = poll(index)
        listed.append((time.monotonic(), {str(path) for path in snapshot.by_path()}))
        return snapshot

    def constructing(executor, run, *, item=None, **kwargs):
        if item is not None:
            admitted.append((time.monotonic(), [member.path for member in item.graph.stamp.members]))
        return construct(executor, run, item=item, **kwargs)
    monkeypatch.setattr(DirectoryIndex, 'poll', listing)
    monkeypatch.setattr(StandardRunExecutor, '_construct', constructing)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, PILATUS), request_value=7134)
    events = []
    try:
        _wait(executor, events, _count_labels('calibrant_scan001', 3), 60.)
        _settle(executor, events, .5)
        started = time.monotonic()
        spec.frame(1, 3)
        _wait(executor, events, _count_labels('calibrant_scan001', 4), 30.)
        for k in range(4, 14):
            spec.frame(1, k)
        _wait(executor, events, _count_labels('calibrant_scan001', 14), 30.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    seen, early = {str(raw / f'calibrant_scan001_{k:04d}.tif') for k in range(3)}, []
    for at, members in admitted:
        early += [path for path in members if at > started and path not in seen
                  and sum(path in paths for done, paths in listed if done < at) < 2]
        seen.update(members)
    assert not early, f'admitted on the listing that first saw them: {early}'
    labels, record = _counted(tmp_path, 'calibrant_scan001')
    assert labels == tuple(range(1, 15)) and _column(record, 'i0') == tuple(100. + k for k in range(14))
    with h5py.File(tmp_path / 'processed' / 'calibrant_scan001_int1d.nexus', 'r') as handle:
        sources = [handle[f'entry/frames/frame_{label:04d}/source/path'][()].decode() for label in labels]
    assert sources == [f'calibrant_scan001_{k:04d}.tif' for k in range(14)], sources


def test_r5c_a_live_series_signs_the_saturation_its_selected_image_records(tmp_path, monkeypatch):
    """Automatic saturation masking: a Live Image Series (TIFF members, or a selected Eiger master) signs the
    ceiling its selected image or container records, as the finite series does, not the Folder rule."""
    import tifffile
    raw = tmp_path / 'raw'
    raw.mkdir()
    for k in range(2):
        tifffile.imwrite(raw / f'cut_{k:04d}.tif', np.full(PILATUS, k, dtype=np.uint32),
                         description='# Count_cutoff 400 counts\n')
    eiger = _Eiger(raw, PILATUS, 2, chunk=1, stem='eiger')
    with h5py.File(eiger.master, 'a') as handle:
        handle['entry/instrument/detector/saturation_value'] = 900
    signed, admitted = [], output_preflight._admitted_saturation
    monkeypatch.setattr(output_preflight, '_admitted_saturation',
                        lambda *args: (signed.append(admitted(*args)), signed[-1])[1])
    for value, (selected, stem) in enumerate((('cut_0000.tif', 'cut'), ('eiger_master.h5', 'eiger'))):
        source = image_series_spec(raw / selected, metadata_format=None)
        finite = graph_saturation(qualify_source_execution_graph(source), detector=None)
        pending = tmp_path / 'pending.h5'
        if value:
            eiger.chunks[0].rename(pending)  # the master records its ceiling before its chunk lands
        executor, events = StandardRunExecutor(join_timeout=2.), []
        identity = _start(executor, _intent(tmp_path, raw, PILATUS, source=source), request_value=7170 + value)
        try:
            if value:
                _wait(executor, events, lambda _values: len(signed) > value, 10.)
                pending.rename(eiger.chunks[0])
            _wait(executor, events, _count_labels(stem, 2), 60.)
        finally:
            terminal = _stop(executor, identity, events)
        assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
        assert (finite.value, finite.source, signed[-1]) == ((400, 900)[value], 'recorded', (400, 900)[value]), (
            finite, signed)


# --- R5: entry modes the maintainer named ----------------------------------------------------------------------

#: R5a: selected file, output stem, frames at Start, metadata policy, and the column (and values) its four frames give.
R5A = {'spec': ('calibrant_scan001_0000.tif', 'calibrant_scan001', 3, 'spec', 'i0', (100., 101., 102., 103.)),
       'txt': ('scan_0000.tif', 'scan', 3, 'txt', 'I0', (0., 1., 2., 3.)),
       'single': ('lone.tif', 'lone', 1, None, None, None), 'container': ('eiger_master.h5', 'eiger', 2, None, None, None)}


@pytest.mark.parametrize('case', list(R5A))
def test_r5a_live_on_an_image_series_follows_that_series_only(tmp_path, case):
    """A numbered series gains only its own numbered members (not another scan, suffix or name); TIFF+txt waits
    for each member's sidecar; an unnumbered image, or a selected Eiger master as it grows, is exactly that file."""
    raw = tmp_path / 'raw'
    raw.mkdir()
    selected, stem, first, fmt, key, values = R5A[case]
    spec, cars = _Spec(raw, PILATUS), _Sidecar(raw, PILATUS)
    spec.scan(1)
    for k in range(3):
        spec.frame(1, k) if case == 'spec' else cars.frame(k)
    eiger = _Eiger(raw, PILATUS, 2, chunk=1, stem='eiger') if case == 'container' else _tiff(raw / 'lone.tif', 7, PILATUS)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, PILATUS, source=image_series_spec(raw / selected, metadata_format=fmt)),
                      request_value=7140 + list(R5A).index(case))
    events = []
    try:
        _wait(executor, events, _count_labels(stem, first), 60.)
        _settle(executor, events, .5)
        if case == 'spec':
            spec.frame(1, 3), spec.scan(2), spec.frame(2, 0), _raw(raw / 'calibrant_scan001_0004.raw', 9, PILATUS)
        elif case == 'txt':
            cars.image(3)
            _settle(executor, events, .5)
            assert len(set(_labels(events, stem))) == 3, 'a member was reduced before its sidecar'
            cars.sidecar(3)
        elif case == 'single':
            _tiff(raw / 'lone_0001.tif', 8, PILATUS)
        else:
            eiger.grow(), _Eiger(raw, PILATUS, 1, stem='later')
        _wait(executor, events, _count_labels(stem, first + (case != 'single')), 20.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    labels, record = _counted(tmp_path, stem)
    assert labels is not None and len(labels) == first + (case != 'single'), labels
    assert key is None or _column(record, key) == values, _column(record, key)
    others = {'calibrant_scan001', 'calibrant_scan002', 'scan', 'lone', 'eiger', 'later'} - {stem}
    assert not [other for other in others if _counted(tmp_path, other)[0] is not None]


R5B = ('grow', 'rows', 'malformed', 'drift', 'reread', 'stop')


@pytest.mark.parametrize('case', R5B)
def test_r5b_spec_with_an_eiger_master_binds_each_frame_to_its_own_row(tmp_path, monkeypatch, case):
    """Frame k of {spec}_scanN_master is row k of scan N, across two external chunks as the last one grows: the
    master waits until each frame has its row (a later row waits for its frame), a malformed row fails loudly, a
    published row rewritten stops the output, a new row rewritten as construction starts is read again, and Stop
    during the arrival's SPEC read ends the Run within 1 s. Each frame is reduced at its own row's energy; the
    frames' source carries the rows' columns as its motors, and the SPEC file is in the source snapshots."""
    raw, stem = tmp_path / 'raw', 'calibrant_scan001'
    raw.mkdir()
    spec, eiger = _Spec(raw, PILATUS, energy=True), _Eiger(raw, PILATUS, 4, chunk=2, stem=stem)
    spec.scan(1)
    for k in range(3 if case == 'rows' else 4):
        spec.row(1, k)
    work, read, entered, closed = _Work(monkeypatch, counters=False), metadata._read_spec_scan, threading.Event(), None
    construct, bind, bound = StandardRunExecutor._construct, NexusStackSource.bind_metadata_provider, []
    monkeypatch.setattr(NexusStackSource, 'bind_metadata_provider',
                        lambda source, provider: (bound.append(provider), bind(source, provider))[1])

    def held(*args, cancelled=None):  # the arrival's own SPEC read waits for Stop
        if cancelled is not None:
            entered.set()
            while not cancelled():
                time.sleep(.01)
        return read(*args, cancelled=cancelled)

    def rewritten(executor, run, **kwargs):  # the new row changes as its construction starts
        monkeypatch.setattr(StandardRunExecutor, '_construct', construct)
        _same_size_rewrite(spec.path, '0.4 104 ', '0.4 904 ')
        return construct(executor, run, **kwargs)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, PILATUS, suffix='_master.h5', fmt='spec'),
                      request_value=7144 + R5B.index(case))
    events = []
    try:
        if case == 'rows':
            _settle(executor, events, 1.)
            assert not _labels(events, stem), 'frames were reduced before every one had its row'
            spec.row(1, 3), spec.row(1, 4)  # the extra row waits for its frame
        _wait(executor, events, _count_labels(stem, 4), 60.)
        _settle_reuse(executor, identity, work, events, 4, time.monotonic() + 60.)
        settled = executor.processed_live_revisions(identity)
        differential = _differential(settled)
        if case == 'malformed':
            with spec.path.open('a') as stream:
                stream.write('0.4 104 not-a-number\n')
        elif case == 'drift':
            _same_size_rewrite(spec.path, '0.1 101 ', '0.1 109 ')
        elif case in ('stop', 'reread'):
            monkeypatch.setattr(*((metadata, '_read_spec_scan', held) if case == 'stop' else
                                  (StandardRunExecutor, '_construct', rewritten)))
        if case not in ('rows', 'malformed'):
            spec.row(1, 4)
        eiger.grow()
        if case == 'stop':
            assert entered.wait(30.), 'the arrival never read its SPEC rows'
            closed = _timed_stop(executor, identity, events, work)
        else:
            _wait(executor, events, lambda values: _terminal(values) or _said(values, 'stopped growing')
                  or len(set(_labels(values, stem))) >= 5, 30.)
            _settle(executor, events, .5)
    finally:
        terminal = closed[3] if closed is not None else _stop(executor, identity, events)
    labels, record = _counted(tmp_path, stem)
    if case == 'malformed':
        assert [kind for kind, _detail in terminal][-1:] == ['FAILED'] and 'row 5' in str(terminal), terminal
    else:
        assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert closed is None or closed[2] <= 1., closed
    assert labels == tuple(range(4 if case in ('malformed', 'drift', 'stop') else 5)), labels
    assert case != 'drift' or _said(events, 'stopped growing')
    assert _column(record, 'i0') == (100., 101., 102., 103., 904. if case == 'reread' else 104.)[:len(labels)]
    assert _column(record, 'c2') == (2., 3., 4., 5., 6.)[:len(labels)]
    assert _column(record, 'del') == pytest.approx((0., .1, .2, .3, .4)[:len(labels)], rel=1e-6, abs=0)
    selection = read_provenance(tmp_path / 'processed' / f'{stem}_int1d.nexus')['config']['energy_selection']
    energies = dict(zip(selection['frame_labels'], selection['frame_energies_eV']))
    assert len(set(energies.values())) == 4 and energies == {k: 1000. * (21 + k) for k in energies}, selection
    assert not differential, differential
    snapshots = output_preflight.source_snapshots(settled[0].decision.item)
    assert snapshots[str(spec.path)]['source_role'] == 'image_metadata', snapshots
    motors = bound[-1].motors()
    assert set(motors) == set(spec.labels), motors
    assert all(list(motors[name]) == [bound[-1].metadata_for(k)[name] for k in range(bound[-1].frame_count())]
               for name in motors), motors


# --- R6: Stop within 1 s through cleanup ------------------------------------------------------------------------

def _timed_stop(executor, identity, events, work):
    """(t1, t2, t3) seconds after t0 = just before stop(), and the terminals: t1 the worker left its work
    (None if it never started), t2 STOPPED drained, t3 close() returned the exact CLEANED receipt."""
    t0 = time.monotonic()
    executor.stop(identity)
    _wait(executor, events, _terminal, 120.)
    t2 = time.monotonic()
    closed = executor.close(identity)
    t3 = time.monotonic()
    assert closed == ExecutorClosed(identity, CleanupStatus.CLEANED), closed
    exits = [at for at, _detail in work.times('work_exit', t0)]
    return (exits[0] - t0 if exits else None), t2 - t0, t3 - t0, _terminals(events)


def _stop_line(label, t1, t2, t3):
    return f'STOP {label} t1={"-" if t1 is None else f"{t1:.3f}"} t2={t2:.3f} t3={t3:.3f}'


@pytest.mark.parametrize('moment', ['idle', 'pending', 'publication'])
def test_r6a_stop_ends_a_small_live_run_within_one_second(tmp_path, monkeypatch, moment):
    raw = tmp_path / 'raw'
    raw.mkdir()
    spec = _Spec(raw, PILATUS)
    spec.scan(1)
    for k in range(4):
        spec.frame(1, k)
    work = _Work(monkeypatch, counters=False)
    entered = work.entered.setdefault('drain_start' if moment == 'publication' else 'poll', threading.Event())
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, PILATUS),
                      request_value=7150 + ('idle', 'pending', 'publication').index(moment))
    events, closed = [], None
    try:
        _wait(executor, events, _count_labels('calibrant_scan001', 4), 60.)
        _settle_reuse(executor, identity, work, events, 4, time.monotonic() + 60.)
        if moment == 'pending':
            spec.image(1, 4)
            work.await_polls(time.monotonic(), 3, time.monotonic() + 30.)
        entered.clear()
        if moment == 'publication':
            for k in range(4, 24):
                spec.frame(1, k)
        assert entered.wait(60.), f'the {moment} moment was never reached'
        left, reached, closed, terminal = _timed_stop(executor, identity, events, work)
    finally:
        if not _terminal(events):
            _stop(executor, identity, events)
    print(_stop_line(moment, left, reached, closed))
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert closed <= 1., f'Stop at {moment}: STOPPED after {reached:.3f} s, closed after {closed:.3f} s'


def test_r6b_stop_while_a_5000_row_spec_arrival_is_qualified_ends_within_one_second(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    spec = _Spec(raw, TINY)
    spec.scan(1)
    for k in range(5000):
        spec.frame(1, k)
    work = _Work(monkeypatch)
    entered = work.entered.setdefault('spec_rows', threading.Event())
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, TINY), request_value=7153)
    events = []
    try:
        assert entered.wait(900.), 'the 5000-row qualification did not start within 15 minutes'
        left, reached, closed, terminal = _timed_stop(executor, identity, events, work)
    finally:
        if not _terminal(events):
            _stop(executor, identity, events)
    print(_stop_line('spec5000', left, reached, closed))
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert closed <= 1., f'STOPPED after {reached:.3f} s, closed after {closed:.3f} s'


# --- R7: lifetime and parity -------------------------------------------------------------------------------------

def test_r7a_a_second_run_qualifies_afresh_and_appends_after_the_first(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    cars = _Sidecar(raw, PILATUS)
    for k in range(4):
        cars.frame(k)
    work = _Work(monkeypatch)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, PILATUS, fmt='txt'), request_value=7160)
    events = []
    try:
        _wait(executor, events, _count_labels('scan', 4), 60.)
        _settle_reuse(executor, identity, work, events, 4, time.monotonic() + 60.)
    finally:
        assert [kind for kind, _detail in _stop(executor, identity, events)] == ['STOPPED']
    cars.frame(4)
    mark = time.monotonic()
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, PILATUS, fmt='txt', mode='Append'), request_value=7161)
    events = []
    try:
        _wait(executor, events, lambda values: 5 in _labels(values, 'scan') or _terminal(values), 60.)
        _settled(executor, identity, 5, time.monotonic() + 60.)
        proved = {detail for _at, detail in work.times('qualify_read', mark)}
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert {str(raw / f'scan_{k:04d}.tif') for k in range(5)} <= proved, 'the new Run proved its inputs itself'
    labels, record = _counted(tmp_path, 'scan')
    assert (labels, _column(record, 'I0')) == ((1, 2, 3, 4, 5), (0., 1., 2., 3., 4.))


def _gi_sidecar(i0, th):
    return (f'# Counters\nI0 = {i0}\n# Motors\nth = {th:.1f}, energy = 16\n'
            'User: energy, time: Mon Jan 15 10:30:00 2024  # Temp\n')


def test_r7b_gi_live_keeps_each_frames_own_incidence_motor(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    for k in range(3):
        (raw / f'scan_{k:04d}.txt').write_text(_gi_sidecar(k, 0.1 * (k + 1)))
        _tiff(raw / f'scan_{k:04d}.tif', k, PILATUS)
    work = _Work(monkeypatch, counters=False)
    executor = StandardRunExecutor(join_timeout=2.)
    gi = GIIntent(enabled=True, incidence_motor='th')
    identity = _start(executor, _intent(tmp_path, raw, PILATUS, fmt='txt', gi=gi), request_value=7162)
    events = []
    try:
        _wait(executor, events, _count_labels('scan_gi', 3), 60.)
        _settle_reuse(executor, identity, work, events, 3, time.monotonic() + 60.)
        (raw / 'scan_0003.txt').write_text(_gi_sidecar(3, 0.4))
        _tiff(raw / 'scan_0003.tif', 3, PILATUS)
        _wait(executor, events, _count_labels('scan_gi', 4), 30.)
        _settle(executor, events, .3)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    # Scan metadata is stored as float32.
    assert _column(_counted(tmp_path, 'scan_gi')[1], 'th') == tuple(float(np.float32(v)) for v in (.1, .2, .3, .4))


# --- P1: whole-poll overhead per poll type at 100, 500 and 1500 points (bars frozen before terminal) ----------

PERF = ('spec_tiff', 'txt', 'nexus', 'eiger', 'aggregate')
#: The GUI's default bins (1D 1000; 2D 1000 x 360): the containers carry the epoch-commit/finish split.
REALISTIC = {'npt': 1000, 'method': 'csr'}, {'npt_rad': 1000, 'npt_azim': 360, 'method': 'csr'}


def _bars(idle, arrival_1, arrival_10, latency, pending=None):
    return {'idle': idle, **({} if pending is None else {'pending': pending}), 'arrival_1': arrival_1,
            'arrival_10': arrival_10, 'arrival_latency': (latency, latency), 'stop_closed': (1., 1.)}


#: Overhead bars (p50 s, p95 s), frozen before terminal (docs c810658f) from each row's own native leaf (leaves/cal,
#: cal3, m2b/d1-p1-500): max(checkpoint bar, 1.5 x measured), rounded up to 0.01 s; 100/500 idle and pending keep
#: p50 .020, p95 max(.040, 1.5 x measured). Latency (write to first construct) gates its p95; SPEC+TIFF 500's leaf
#: predates d2c853da, so one poll is added before the margin. Stop: 1 s. Containers have no pending phase: N/A.
BARS = {
    ('spec_tiff', 100): _bars(idle=(.02, .04), pending=(.02, .04), arrival_1=(.20, .22), arrival_10=(1.01, 1.04), latency=.49),
    ('spec_tiff', 500): _bars(idle=(.02, .05), pending=(.02, .06), arrival_1=(.54, .65), arrival_10=(1.55, 1.70), latency=.63),
    ('spec_tiff', 1500): _bars(idle=(.02, .06), pending=(.07, .10), arrival_1=(.78, .96), arrival_10=(2.25, 2.41), latency=.94),
    ('txt', 100): _bars(idle=(.02, .04), pending=(.02, .04), arrival_1=(.14, .20), arrival_10=(.34, .50), latency=.42),
    ('txt', 500): _bars(idle=(.02, .09), pending=(.02, .11), arrival_1=(.34, .41), arrival_10=(.72, .79), latency=.59),
    ('txt', 1500): _bars(idle=(.02, .13), pending=(.07, .14), arrival_1=(.76, .95), arrival_10=(1.70, 2.89), latency=.84),
    ('nexus', 100): _bars(idle=(.02, .04), arrival_1=(.18, .20), arrival_10=(.72, .73), latency=.30),
    ('nexus', 500): _bars(idle=(.02, .04), arrival_1=(.47, .52), arrival_10=(1.57, 1.64), latency=.35),
    ('nexus', 1500): _bars(idle=(.02, .06), arrival_1=(1.04, 1.05), arrival_10=(3.07, 3.09), latency=.60),
    ('eiger', 100): _bars(idle=(.02, .04), arrival_1=(.25, .26), arrival_10=(.46, .50), latency=.35),
    ('eiger', 500): _bars(idle=(.02, .04), arrival_1=(.35, .37), arrival_10=(.64, .66), latency=.41),
    ('eiger', 1500): _bars(idle=(.02, .06), arrival_1=(.61, .63), arrival_10=(1.08, 1.11), latency=.60),
    ('aggregate', 100): _bars(idle=(.02, .04), pending=(.02, .07), arrival_1=(.12, .20), arrival_10=(.30, .50), latency=.47),
    ('aggregate', 500): _bars(idle=(.02, .05), pending=(.02, .10), arrival_1=(.15, .25), arrival_10=(.37, .60), latency=.59),
    ('aggregate', 1500): _bars(idle=(.03, .09), pending=(.03, .19), arrival_1=(.26, .50), arrival_10=(.60, 1.00), latency=.90),
}


class _Feed:
    """The next frame of a performance source: complete, or its image now and its metadata later.

    SPEC+TIFF and SPEC+Eiger carry 20 MB of earlier-scan history; the aggregate's SPEC file holds only its own
    scans. NeXus runs 1D+2D and SPEC+Eiger 1D (frame k is row k; a row lands before its frame), both on a
    Pilatus-sized detector with the GUI's default bins."""

    def __init__(self, name, raw, points):
        self.name, self.k, self.slot = name, 0, '_int2d' if name == 'nexus' else '_int1d'
        if name in ('spec_tiff', 'aggregate'):
            self.owner = _Spec(raw, TINY, history_bytes=20 << 20 if name == 'spec_tiff' else 0)
            scans = 10 if name == 'aggregate' else 1
            for scan in range(1, scans + 1):
                self.owner.scan(scan)
                for k in range(points // scans):
                    self.owner.frame(scan, k)
            self.scan = scans + 1 if name == 'aggregate' else 1
            if name == 'aggregate':
                self.owner.scan(self.scan)
                self.owner.frame(self.scan, 0)
                self.k = 1
            else:
                self.k = points
            self.stem, self.first = f'calibrant_scan{self.scan:03d}', 1
        elif name == 'txt':
            self.owner, self.k, self.stem, self.first = _Sidecar(raw, TINY), points, 'scan', 1
            for k in range(points):
                self.owner.frame(k)
        else:
            self.owner = _Nexus(raw, PILATUS, points) if name == 'nexus' else _Eiger(raw, PILATUS, points,
                                                                                         stem='calibrant_scan001')
            self.stem, self.first = ('scan_0' if name == 'nexus' else 'calibrant_scan001'), 0
            if name == 'eiger':
                self.spec, self.k = _Spec(raw, TINY, history_bytes=20 << 20), points
                self.spec.scan(1)
                for k in range(points):
                    self.spec.row(1, k)

    def frame(self):
        if self.name in ('spec_tiff', 'aggregate'):
            self.owner.frame(self.scan, self.k)
        elif self.name == 'txt':
            self.owner.frame(self.k)
        else:
            if self.name == 'eiger':
                self.spec.row(1, self.k)
            self.owner.grow()
        self.k += 1

    def pending(self):
        if self.name in ('spec_tiff', 'aggregate'):
            self.owner.image(self.scan, self.k)
            return True
        if self.name == 'txt':
            self.owner.image(self.k)
            return True
        return False

    def land(self):
        if self.name in ('spec_tiff', 'aggregate'):
            self.owner.row(self.scan, self.k)
        else:
            self.owner.sidecar(self.k)
        self.k += 1

    def intent(self, tmp_path, raw):
        if self.name == 'nexus':
            return _intent(tmp_path, raw, PILATUS, suffix='.nxs', fmt=None, processing='Int 2D', one=REALISTIC[0],
                           two=REALISTIC[1])
        if self.name == 'eiger':
            return _intent(tmp_path, raw, PILATUS, suffix='_master.h5', fmt='spec', one=REALISTIC[0])
        return _intent(tmp_path, raw, TINY, fmt='txt' if self.name == 'txt' else 'spec')

    def published(self, events):
        """The frame labels this source's output has published."""
        return {event.frame_key.local_frame_label for event in events
                if event.kind is StandardEventKind.FRAME_READY and f'{self.stem}{self.slot}' in str(event.artifact)}

    def count(self, total):
        return lambda values: len(self.published(values)) >= total or _terminal(values)


def _quantiles(values):
    ordered = sorted(values)
    return statistics.median(ordered), ordered[min(len(ordered) - 1, int(round(.95 * (len(ordered) - 1))))]


@pytest.mark.parametrize('points', [100, 500, 1500])
@pytest.mark.parametrize('name', PERF)
def test_p1_whole_poll_overhead_per_poll_type(tmp_path, monkeypatch, record_property, name, points):
    raw = tmp_path / 'raw'
    raw.mkdir()
    built = time.monotonic()
    feed = _Feed(name, raw, points)
    record_property('fixture_build_s', round(time.monotonic() - built, 3))
    record_property('detector_shape', 'x'.join(map(str, PILATUS if name in ('nexus', 'eiger') else TINY)))
    work = _Work(monkeypatch, counters=False)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, feed.intent(tmp_path, raw), request_value=7200 + 10 * PERF.index(name)
                      + (500, 1500, 100).index(points))
    events, measured, phase, terminal, problems, footprint = [], {}, 'initial', [], [], None
    budget = 900.
    deadline, started = time.monotonic() + budget, time.monotonic()
    initial = 1 if name == 'aggregate' else (points if name not in ('nexus', 'eiger') else points)
    pending = name not in ('nexus', 'eiger')
    try:
        _wait(executor, events, feed.count(initial), budget)
        record_property('initial_admission_and_reduction_s', round(time.monotonic() - started, 3))
        print(f'P1 {name} {points} initial admission and reduction {time.monotonic() - started:.1f}s')
        phase = 'idle'
        polls = work.await_polls(time.monotonic(), 12, deadline)
        measured['idle'] = [work.overhead(a, b) for a, b in zip(polls[1:11], polls[2:12])]
        print(f'P1 {name} {points} idle polls (s, listings)', [(round(work.overhead(a, b), 4), len(work.times('discovery', a, b))) for a, b in zip(polls[1:11], polls[2:12])])
        # Every real listing in the idle span (a skipped poll reports none); no gap between listings over 1 s.
        listed = work.times('discovery', polls[1], polls[11])
        measured['idle_discovery_call'] = [took for _at, took in listed]
        record_property('idle_listings', len(listed))
        ends = [at for at, _took in work.times('discovery', 0., polls[1])][-1:] + [at for at, _took in listed]
        assert ends and max(b - a for a, b in zip(ends, [*ends[1:], polls[11]])) <= 1., 'idle listing gap'
        phase = 'pending'
        if pending:
            feed.pending()
            polls = work.await_polls(time.monotonic(), 12, deadline)
            measured['pending'] = [work.overhead(a, b) for a, b in zip(polls[1:11], polls[2:12])]
            print(f'P1 {name} {points} pending polls (s, listings)', [(round(work.overhead(a, b), 4), len(work.times('discovery', a, b))) for a, b in zip(polls[1:11], polls[2:12])])
            target = len(feed.published(events)) + 1
            feed.land()
            _wait(executor, events, feed.count(target), max(1., deadline - time.monotonic()))
            assert len(feed.published(events)) == target, 'pending: frame progress'
        for key in ('commit_epoch_net', 'reduction_excluded', 'arrival_latency'):
            measured[key] = []
        for burst, key, repeats in ((1, 'arrival_1', 10), (10, 'arrival_10', 3)):
            phase, measured[key] = key, []
            for _repeat in range(repeats):
                mark = time.monotonic()
                target = len(feed.published(events)) + burst
                for _index in range(burst):
                    feed.frame()
                _wait(executor, events, feed.count(target), max(1., deadline - time.monotonic()))
                assert len(feed.published(events)) == target, f'{key}: frame progress'
                constructs = [at for at, _detail in work.times('construct', mark)]
                # The poll that admits the frames may have started before they were written.
                first = [at for at in work.polls() if at <= constructs[0]][-1]
                after = work.await_polls(constructs[-1], 1, deadline)[0]
                commits = work.net('commit_epoch', first, after)
                assert commits, f'{key}: no epoch commit'
                measured[key].append(work.overhead(first, after))
                measured['commit_epoch_net'] += commits
                measured['reduction_excluded'].append(sum(took for _at, took in work.times('drain', first, after)))
                measured['arrival_latency'].append(constructs[0] - mark)
        # The growing output only: later scans append to a settled scan's SPEC file after its qualification.
        grown = [value for value in executor.processed_live_revisions(identity) if not value.partial and (
            name in ('nexus', 'eiger') or value.decision.item.graph.group_key == feed.stem)]
        assert grown, 'the growing output has no settled attempt'
        problems = _differential(grown)
        if points == 1500:
            footprint = _reuse_footprint(executor, identity, events)
        phase = 'stop'
        work.await_polls(time.monotonic(), 2, deadline)
        stopping = time.monotonic()
        left, reached, closed, terminal = _timed_stop(executor, identity, events, work)
        record_property('stop_t1_s', None if left is None else round(left, 4))
        record_property('stop_t2_s', round(reached, 4))
        record_property('stop_cleanup_s', round(sum(took for _at, took in work.times('cleanup', stopping)), 4))
        measured['stop_closed'] = [closed]
        measured['finish'] = work.net('finish', stopping, time.monotonic())
    except AssertionError:
        record_property('deadline_phase', phase)
        print(f'P1 {name} {points} failed in {phase}; terminals {_terminals([*events, *executor.drain_events()])}')
        raise
    finally:
        if not _terminal(events):
            terminal = _stop(executor, identity, events)
        for key, values in measured.items():
            if not values:
                continue
            p50, p95 = _quantiles(values)
            record_property(f'{key}_p50_s', round(p50, 4))
            record_property(f'{key}_p95_s', round(p95, 4))
            print(f'P1 {name} {points} {key} n={len(values)} p50={p50:.4f}s p95={p95:.4f}s')
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert not problems, problems
    counts = {key: len(values) for key, values in measured.items()}
    assert counts == {'idle': 10, 'idle_discovery_call': counts['idle_discovery_call'], **({'pending': 10} if pending
                      else {}), 'commit_epoch_net': counts['commit_epoch_net'], 'reduction_excluded': 13,
                      'arrival_latency': 13, 'arrival_1': 10, 'arrival_10': 3, 'stop_closed': 1, 'finish': 1}, counts
    assert counts['commit_epoch_net'] >= 13, counts
    assert len(feed.published(events)) == initial + int(pending) + 40, 'frame progress'
    if footprint is not None:
        per_frame, fixed, accepted = footprint
        record_property('reuse_bytes_per_frame', round(per_frame / accepted, 1))
        record_property('reuse_fixed_bytes', fixed)
        assert per_frame <= 4096 * accepted and per_frame + fixed <= 20 << 20, footprint
    failed, bars = {}, BARS[(name, points)]
    assert set(bars) == {'idle', 'arrival_1', 'arrival_10', 'arrival_latency', 'stop_closed', *['pending'] * pending}
    assert set(bars) <= set(measured), f'barred phases never measured: {set(bars) - set(measured)}'
    for key, (bar50, bar95) in bars.items():
        p50, p95 = _quantiles(measured[key])
        record_property(f'{key}_bar_p50_s', bar50)
        record_property(f'{key}_bar_p95_s', bar95)
        if p50 > bar50 or p95 > bar95:
            failed[key] = (round(p50, 4), round(p95, 4), bar50, bar95)
    # The bars are guidelines: the maintainer chose report-only on hosted CI (CI set) and enforced locally.
    enforced = 'CI' not in os.environ
    record_property('bars_failed', failed)
    record_property('bars_enforced', enforced)
    if enforced:
        assert not failed, failed


def _reuse_footprint(executor, identity, events):
    """(per-frame bytes, fixed bytes, accepted frames) of the Run's retained reuse state, summed flat over its
    records: tuples of primitive values whose first value is the frame they serve, or None if fixed."""
    records = executor.live_reuse_records(identity)
    kinds = {type(value) for record in records for value in record}
    assert kinds <= {str, int, float, bool, type(None)}, f'retained reuse state holds {kinds}'
    sizes = [(record[0] is None, sys.getsizeof(record) + sum(map(sys.getsizeof, record))) for record in records]
    accepted = len({(str(event.artifact), event.frame_key.local_frame_label) for event in events
                    if event.kind is StandardEventKind.FRAME_READY})
    return sum(size for fixed, size in sizes if not fixed), sum(size for fixed, size in sizes if fixed), accepted
