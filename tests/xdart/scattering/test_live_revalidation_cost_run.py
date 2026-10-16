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
import threading
import time

import fabio
import h5py
import numpy as np
import pytest
import silx.io.specfile as silx_specfile

import xrd_tools.io.metadata as metadata
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
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xrd_tools.io.nexus import read_scan_metadata
from xrd_tools.reduction.core import ReductionSession
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xrd_tools.sources.base import BaseFrameSource
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xrd_tools.sources.execution_graph import SourceFileState, qualify_source_execution_graph
from xrd_tools.sources.selection import DirectorySourceSpec, image_series_spec

POLL = run_executor._LIVE_DIRECTORY_POLL_S
CONTENT = ('qualify_read', 'reduce_read', 'spec_scan_read', 'spec_rows', 'silx_open', 'hdf5_open',
           'graph_validation')
PILATUS, TINY = (195, 487), (16, 16)
LABELS = ('del', 'i0', *(f'c{j}' for j in range(2, 22)))
SPEC_HEAD = '#F calibrant\n#E 1\n#O0 th  chi\n\n'
#: Bars frozen at the checkpoint from the parent measurements: (family, points) -> {phase: (p50 s, p95 s)}.
BARS: dict[tuple[str, int], dict[str, tuple[float | None, float | None]]] = {}


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
        monkeypatch.setattr(StandardRunExecutor, '_execute_live_directory', live)
        monkeypatch.setattr(DirectoryIndexSession, 'observe', observe)
        monkeypatch.setattr(ReductionSession, 'drain', drain)
        monkeypatch.setattr(StandardRunExecutor, '_construct', construct)
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
        """One poll's overhead: its interval less the loop's wait and the new frames' reduction."""
        return end - start - POLL - sum(detail for _at, detail in self.times('drain', start, end))


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
    """SPEC + TIFF or raw images, SSRL names, 22 columns; i0 of point k of scan s is 100 * s + k."""

    def __init__(self, raw, shape, suffix='.tif', history_bytes=0):
        self.raw, self.shape, self.suffix, self.path = raw, shape, suffix, raw / 'calibrant'
        parts, size, scan = [SPEC_HEAD], len(SPEC_HEAD), 1000
        while size < history_bytes:  # earlier scans the Live Run does not reduce: no image names them
            scan += 1
            text = self._header(scan) + ''.join(self._line(scan, k) for k in range(400))
            parts.append(text)
            size += len(text)
        self.path.write_text(''.join(parts))

    @staticmethod
    def _header(scan):
        return f'#S {scan} ascan th 0 1 10 1\n#P0 0.5 1\n#N {len(LABELS)}\n#L ' + '  '.join(LABELS) + '\n'

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

    def __init__(self, raw, shape, frames, chunk=100):
        self.raw, self.shape, self.master, self.chunks = raw, shape, raw / 'scan_master.h5', []
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
        path = self.raw / f'scan_data_{len(self.chunks) + 1:06d}.h5'
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('entry/data/data', data=np.ones((frames, *self.shape), dtype=np.uint16),
                                  maxshape=(None, *self.shape), chunks=(1, *self.shape))
        self.chunks.append(path)

    def grow(self, count=1):
        with _open_for_write(self.chunks[-1], 'a') as handle:
            data = handle['entry/data/data']
            data.resize((data.shape[0] + count, *self.shape))
            data[-count:] = 2


def _intent(tmp_path, raw, shape, *, suffix='.tif', fmt='spec', mode='Overwrite', gi=None, source=None):
    return RunIntent(
        source_spec=source or DirectorySourceSpec(raw, suffixes=(suffix,), metadata_format=fmt),
        poni_file=str(_poni(tmp_path, shape)), project_root=str(raw), save_path=str(tmp_path / 'processed'),
        output_mode=mode, processing_mode='Int 1D', live_mode=True, max_cores=1,
        bai_1d_args={'npt': 8, 'method': 'numpy'}, **({} if gi is None else {'gi': gi}))


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
    """Every settled TIFF-series graph equals a fresh full qualification, in-memory facts included."""
    problems = []
    for attempt in attempts:
        graph = attempt.decision.item.graph
        if graph.stamp.adapter_id != 'tiff_series':
            continue
        fresh = qualify_source_execution_graph(
            graph.execution_source, infer_point_images=graph.infers_point_images,
            published_members=len(graph.stamp.members))
        if fresh != graph:
            problems.append(f'{graph.group_key}: graph differs from a fresh qualification')
        if fresh.infers_point_images != graph.infers_point_images:
            problems.append(f'{graph.group_key}: infers_point_images differs')
        for mine, theirs in zip(graph.stamp.metadata_sources, fresh.stamp.metadata_sources, strict=True):
            if (mine.spec_extent, dict(mine.spec_values or {})) != (theirs.spec_extent,
                                                                    dict(theirs.spec_values or {})):
                problems.append(f'{mine.source_path}: retained SPEC extent/values differ')
    return problems


def _content(found, published=()):
    """Counted content work, and the published members whose metadata was read again."""
    work = {seam: len(found[seam]) for seam in CONTENT if found.get(seam)}
    reread = sorted({detail for seam in ('qualify_read', 'reduce_read') for detail in found.get(seam, ())
                     if detail in published})
    return work, reread


def _report(row, phase, found):
    """Print one phase's counted work (calls per seam; rows, frames and seconds summed) for the record."""
    summed = {'spec_rows', 'to_scan_frames', 'drain'}
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
        (raw / 'scan1_0001.txt').write_text(SIDECARS['txt'](99.))  # a published sidecar changes
        changed = time.monotonic()
        _wait(executor, events, lambda values: _said(values, 'stopped growing') or _terminal(values), 10.)
        noticed = time.monotonic() - changed
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _said(events, 'scan1_0001.txt'), 'the drift names the changed sidecar'
    assert (noticed <= 1. + 2 * POLL, idle) == (True, {}), (round(noticed, 3), idle)


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
    outcome = ([kind for kind, _detail in terminal], _counted(tmp_path, 'scan')[0], _said(events, 'stopped growing'),
               [detail for _kind, detail in terminal])
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


# --- R5: entry modes the maintainer named ----------------------------------------------------------------------

def test_r5a_live_on_an_image_series_follows_that_series_only(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    spec = _Spec(raw, PILATUS)
    spec.scan(1)
    for k in range(3):
        spec.frame(1, k)
    source = image_series_spec(raw / 'calibrant_scan001_0000.tif', metadata_format='spec')
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, PILATUS, source=source), request_value=7140)
    events = []
    try:
        _wait(executor, events, _count_labels('calibrant_scan001', 3), 60.)
        _settle(executor, events, .5)
        spec.frame(1, 3)
        spec.scan(2)
        spec.frame(2, 0)  # another scan in the same folder is not this series
        _wait(executor, events, _count_labels('calibrant_scan001', 4), 20.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _column(_counted(tmp_path, 'calibrant_scan001')[1], 'i0') == (100., 101., 102., 103.)
    assert _counted(tmp_path, 'calibrant_scan002')[0] is None


def test_r5b_spec_with_an_eiger_master_binds_each_frame_to_its_own_row(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    spec = _Spec(raw, PILATUS)
    spec.scan(1)
    for k in range(4):
        spec.row(1, k)
    member, master = raw / 'calibrant_scan001_data_000001.h5', raw / 'calibrant_scan001_master.h5'
    with h5py.File(member, 'w') as handle:
        handle.create_dataset('entry/data/data', data=np.ones((4, *PILATUS), dtype=np.uint16),
                              maxshape=(None, *PILATUS), chunks=(1, *PILATUS))
    with h5py.File(master, 'w') as handle:
        entry = handle.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        data = entry.create_group('data')
        data.attrs['NX_class'] = 'NXdata'
        data['data_000001'] = h5py.ExternalLink(member.name, '/entry/data/data')
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, PILATUS, suffix='_master.h5', fmt='spec'),
                      request_value=7141)
    events = []
    try:
        _wait(executor, events, _count_labels('calibrant_scan001', 4), 60.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _column(_counted(tmp_path, 'calibrant_scan001')[1], 'i0') == (100., 101., 102., 103.)


# --- R6: Stop within 1 s through cleanup ------------------------------------------------------------------------

def _timed_stop(executor, identity, events):
    """(t2 - t0, t3 - t0, terminals): STOPPED drained and close() returned, after stop() was accepted."""
    t0 = time.monotonic()
    executor.stop(identity)
    _wait(executor, events, _terminal, 120.)
    t2 = time.monotonic()
    executor.close(identity)
    return t2 - t0, time.monotonic() - t0, _terminals(events)


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
        reached, closed, terminal = _timed_stop(executor, identity, events)
    finally:
        if not _terminal(events):
            _stop(executor, identity, events)
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
        reached, closed, terminal = _timed_stop(executor, identity, events)
    finally:
        if not _terminal(events):
            _stop(executor, identity, events)
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


# --- P1: whole-poll overhead per poll type at 500 and 5000 points (bars frozen at the checkpoint) --------------

PERF = ('spec_tiff', 'txt', 'nexus', 'eiger', 'aggregate')


class _Feed:
    """The next frame of a performance source: complete, or its image now and its metadata later."""

    def __init__(self, name, raw, points):
        self.name, self.k = name, 0
        if name in ('spec_tiff', 'aggregate'):
            self.owner = _Spec(raw, TINY, history_bytes=20 << 20)
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
            self.owner = _Nexus(raw, TINY, points) if name == 'nexus' else _Eiger(raw, TINY, points)
            self.stem, self.first = ('scan_0' if name == 'nexus' else 'scan'), 0

    def frame(self):
        if self.name in ('spec_tiff', 'aggregate'):
            self.owner.frame(self.scan, self.k)
        elif self.name == 'txt':
            self.owner.frame(self.k)
        else:
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
        if self.name in ('nexus', 'eiger'):
            return _intent(tmp_path, raw, TINY, suffix='.nxs' if self.name == 'nexus' else '_master.h5', fmt=None)
        return _intent(tmp_path, raw, TINY, fmt='txt' if self.name == 'txt' else 'spec')


def _quantiles(values):
    ordered = sorted(values)
    return statistics.median(ordered), ordered[min(len(ordered) - 1, int(round(.95 * (len(ordered) - 1))))]


@pytest.mark.parametrize('points', [500, 5000])
@pytest.mark.parametrize('name', PERF)
def test_p1_whole_poll_overhead_per_poll_type(tmp_path, monkeypatch, record_property, name, points):
    raw = tmp_path / 'raw'
    raw.mkdir()
    built = time.monotonic()
    feed = _Feed(name, raw, points)
    record_property('fixture_build_s', round(time.monotonic() - built, 3))
    record_property('detector_shape', 'x'.join(map(str, TINY)))
    work = _Work(monkeypatch, counters=False)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, feed.intent(tmp_path, raw), request_value=7200 + 10 * PERF.index(name)
                      + (points == 5000))
    events, measured, phase, terminal = [], {}, 'initial', []
    deadline, started = time.monotonic() + 900., time.monotonic()
    initial = 1 if name == 'aggregate' else (points if name not in ('nexus', 'eiger') else points)
    try:
        _wait(executor, events, _count_labels(feed.stem, initial), 900.)
        record_property('initial_admission_and_reduction_s', round(time.monotonic() - started, 3))
        phase = 'idle'
        polls = work.await_polls(time.monotonic(), 12, deadline)
        measured['idle'] = [work.overhead(a, b) for a, b in zip(polls[1:11], polls[2:12])]
        phase = 'pending'
        if feed.pending():
            polls = work.await_polls(time.monotonic(), 12, deadline)
            measured['pending'] = [work.overhead(a, b) for a, b in zip(polls[1:11], polls[2:12])]
            target = len(set(_labels(events, feed.stem))) + 1
            feed.land()
            _wait(executor, events, _count_labels(feed.stem, target), max(1., deadline - time.monotonic()))
        for burst, key, repeats in ((1, 'arrival_1', 10), (10, 'arrival_10', 3)):
            phase, measured[key] = key, []
            for _repeat in range(repeats):
                mark = time.monotonic()
                target = len(set(_labels(events, feed.stem))) + burst
                for _index in range(burst):
                    feed.frame()
                _wait(executor, events, _count_labels(feed.stem, target), max(1., deadline - time.monotonic()))
                constructs = [at for at, _detail in work.times('construct', mark)]
                after = work.await_polls(constructs[-1], 1, deadline)[0]
                before = [at for at in work.polls(mark) if at <= constructs[-1]][-1]
                measured[key].append(work.overhead(before, after))
        phase = 'stop'
        work.await_polls(time.monotonic(), 2, deadline)
        _reached, closed, terminal = _timed_stop(executor, identity, events)
        measured['stop_closed'] = [closed]
    except AssertionError:
        record_property('deadline_phase', phase)
        raise
    finally:
        if not _terminal(events):
            terminal = _stop(executor, identity, events)
        for key, values in measured.items():
            p50, p95 = _quantiles(values)
            record_property(f'{key}_p50_s', round(p50, 4))
            record_property(f'{key}_p95_s', round(p95, 4))
            print(f'P1 {name} {points} {key} n={len(values)} p50={p50:.4f}s p95={p95:.4f}s')
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    failed = {}
    for key, (bar50, bar95) in BARS.get((name, points), {}).items():
        p50, p95 = _quantiles(measured[key])
        if bar50 is not None and p50 > bar50 or bar95 is not None and p95 > bar95:
            failed[key] = (round(p50, 4), round(p95, 4), bar50, bar95)
    assert not failed, failed
