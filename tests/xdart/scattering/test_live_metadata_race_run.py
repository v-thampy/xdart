"""LIVE-METADATA-DISCOVERY-RACE-20261012 frozen live-Run oracle (Claude, October 12): GR1.

A real live Run (StandardRunExecutor on a watched directory whose images carry .txt sidecars). The next frame lands
while the Run's own thread is reading an earlier frame's metadata: a real arrival, made inside that read through the
reader's sidecar lookup the first time a worker thread looks a sidecar up after frame 1 was reduced, one clock tick
after the reader recorded the directory (probe p01).
"""
import threading
import time

import fabio
import numpy as np

import xrd_tools.io.metadata as metadata
import xrd_tools.sources.registry  # noqa: F401  # register built-in readers
from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _drain_until, _start
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec

SIDECAR = '# Counters\nI0 = 1\n# Motors\nenergy = 16\nUser: energy, time: Mon Jan 15 10:30:00 2024  # Temp\n'


def _arrive(directory, index):
    """One frame of an acquisition: its sidecar, then its image renamed into place whole."""
    image = directory / f'scan_{index:04d}.tif'
    image.with_suffix('.txt').write_text(SIDECAR)
    partial = directory / f'scan_{index:04d}.part'
    fabio.tifimage.TifImage(data=np.full((195, 487), index, dtype=np.uint16)).write(str(partial))
    partial.replace(image)


def _reregister_image_adapter(monkeypatch, image):
    """Register the image adapter again under its own id.

    A Live Run re-reads a settled output's metadata only when something changed;
    a registration change is one, so the Run's next re-check is a strict read.
    """
    from dataclasses import replace
    from xrd_tools.sources import adapters
    entry = adapters._ADAPTERS[adapters.candidate_owner(image).id]
    monkeypatch.setitem(adapters._ADAPTERS, entry.adapter.id, adapters._Entry(
        replace(entry.adapter), entry.builtin, entry.seq + 100000))


def _construct_strictly(monkeypatch, image, members):
    """Make the construction of a group of ``members`` images validate the whole group again.

    Construction re-checks the proof of what its poll admitted; a registration change as it starts fails that
    proof, so construction validates the group (the strict re-check these rows exercise).
    """
    real = StandardRunExecutor._construct

    def construct(executor, run, **kwargs):
        if kwargs.get('item') is not None and len(kwargs['item'].source_stamp.members) == members:
            _reregister_image_adapter(monkeypatch, image)
        return real(executor, run, **kwargs)
    monkeypatch.setattr(StandardRunExecutor, '_construct', construct)


def _labels(events):
    return {event.frame_key.local_frame_label for event in events if event.kind is StandardEventKind.FRAME_READY}


def _terminal(events):
    return any(event.kind in _TERMINAL for event in events)


def test_gr1_a_live_run_keeps_going_when_a_frame_lands_during_a_metadata_read(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _arrive(raw, 1)
    poni = tmp_path / 'cal.poni'
    write_poni(poni)
    reduced, landed, real = threading.Event(), [], metadata._find_sidecar

    def lookup(path, ext):
        if reduced.is_set() and not landed and threading.current_thread() is not threading.main_thread():
            landed.append(threading.current_thread().name)
            time.sleep(.02)
            _arrive(raw, 2)
        return real(path, ext)
    monkeypatch.setattr(metadata, '_find_sidecar', lookup)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=('.tif',), metadata_format='txt'), poni_file=str(poni),
        project_root=str(raw), save_path=str(tmp_path / 'processed'), output_mode='Overwrite',
        processing_mode='Int 1D', live_mode=True, max_cores=1, bai_1d_args={'npt': 8, 'method': 'numpy'}),
        request_value=2610)
    events = []
    try:
        events += _drain_until(executor, lambda values: 1 in _labels(values), timeout=60.)
        reduced.set()
        _reregister_image_adapter(monkeypatch, raw / 'scan_0001.tif')  # the next re-check reads metadata
        events += _drain_until(executor, lambda values: 2 in _labels(events + list(values)) or _terminal(values),
                               timeout=60.)
        failed = [event for event in events if event.kind is StandardEventKind.FAILED]
        assert not failed, failed[0].detail
        assert landed, "frame 2 landed during a metadata read on one of the Run's own threads"
        assert 2 in _labels(events), 'the Run reduced the frame that landed'
    finally:
        executor.stop(identity)
        if not _terminal(events):
            events += _drain_until(executor, _terminal, timeout=60.)
        executor.close(identity)
    assert [event.kind for event in events if event.kind in _TERMINAL] == [StandardEventKind.STOPPED]
