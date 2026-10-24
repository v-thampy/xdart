"""A Live directory Append leaves a scan it has settled alone until its images change (maintainer,
2026-10-23: "If something has already been processed, it should be skipped anyway"; Codex,
DESIGN-append-image-reprocess-125: a watch, never a permanent skip, and never for the output a Live epoch owns).

Beamline 2026-10-23 (dev 21f4872d, Windows share): a Live Run read every finished scan's sidecars again before its
first frame (over 40 s), then refused finished scans one by one after the beamline touched their files, and,
settling the scan still growing on the way, failed it at its next frame: "cannot bind an allocation to a populated
store"."""
from __future__ import annotations

import os

from tests.xdart.scattering.test_live_qualify_refusal_run import (
    SIDECARS, _frame, _labels, _said, _settle, _stop, _terminals, _wait,
)
from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal, _start
from xdart.gui.tabs.scattering.adapters import run_executor
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


def _intent(tmp_path, raw, *, live):
    poni = tmp_path / 'cal.poni'
    write_poni(poni)
    return RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=('.tif',), metadata_format='txt'), poni_file=str(poni),
        project_root=str(raw), save_path=str(tmp_path / 'processed'), output_mode='Append',
        processing_mode='Int 1D', live_mode=live, max_cores=1, bai_1d_args={'npt': 8, 'method': 'numpy'})


def test_live_leaves_finished_scans_alone_while_another_scan_grows(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    for index in (1, 2):
        _frame(raw, 'done', index, 'txt', float(index))
    executor, identity, events = _run_to_terminal(_intent(tmp_path, raw, live=False), request_value=8801)
    executor.close(identity)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ['FINISHED']

    admitted = []
    real_materialize = run_executor.materialize_live_directory_group

    def materialize(receipt, configuration, session, group, **kwargs):
        admitted.append(group.target.name)
        return real_materialize(receipt, configuration, session, group, **kwargs)

    monkeypatch.setattr(run_executor, 'materialize_live_directory_group', materialize)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, live=True), request_value=8802)
    events = []
    try:
        _frame(raw, 'grow', 1, 'txt', 1.0)
        _wait(executor, events, lambda values: 1 in _labels(values, 'grow') or any(
            event.kind in _TERMINAL for event in values))
        before_touch = len(admitted)  # a first sight is admitted as before (it presents the saved frames)
        # The beamline rewrites a finished scan's sidecar after the scan: its fingerprint moves, its images do not.
        (raw / 'done_0001.txt').write_text(SIDECARS['txt'](9.0))
        _settle(executor, events, 1.)
        _frame(raw, 'grow', 2, 'txt', 2.0)
        _wait(executor, events, lambda values: 2 in _labels(values, 'grow') or any(
            event.kind in _TERMINAL for event in values))
        _settle(executor, events, .5)
        catalog = executor.frame_catalog(identity).entries
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    stopped = next(event for event in events if event.kind is StandardEventKind.STOPPED)
    assert stopped.warnings == () and not _said(events, 'stopped growing'), (stopped.warnings, _terminals(events))
    # Shown at the first poll from its committed record (nothing written yet), never probed or admitted, even
    # after its sidecar moved; its saved frames are in the Run's navigation ahead of the new scan's.
    assert 'done_int1d.nexus' not in admitted and before_touch >= 1, admitted
    assert sorted(set(_labels(events, 'grow'))) == [1, 2]
    assert [(key.source_scan, key.local_frame_label) for key in catalog][:2] == [('done', 1), ('done', 2)], catalog


def _covered_run(tmp_path, monkeypatch, request_value):
    """A finished scan's output, then a Live Run that covers it: (raw, executor, identity, coverage reads, shown,
    stats), the last three filled by call-through spies."""
    raw = tmp_path / 'raw'
    raw.mkdir()
    for index in (1, 2):
        _frame(raw, 'done', index, 'txt', float(index))
    executor, identity, events = _run_to_terminal(_intent(tmp_path, raw, live=False), request_value=request_value)
    executor.close(identity)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ['FINISHED']
    covered, shown, stats = [], [], []
    real_coverage, real_shown = run_executor.live_group_coverage, StandardRunExecutor._present_live_saved_output
    real_stat = getattr(run_executor, '_target_stat', None)

    def coverage(receipt, group):
        result = real_coverage(receipt, group)
        covered.append((group.target.name, result is not None))
        return result

    def show(self, run, group, completed):
        shown.append(group.target.name)
        return real_shown(self, run, group, completed)

    def stat(path):
        stats.append(path.name)
        return real_stat(path)
    monkeypatch.setattr(run_executor, 'live_group_coverage', coverage)
    monkeypatch.setattr(StandardRunExecutor, '_present_live_saved_output', show)
    if real_stat is not None:
        monkeypatch.setattr(run_executor, '_target_stat', stat)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, live=True), request_value=request_value + 1)
    return raw, executor, identity, covered, shown, stats


def test_f3_a_finished_scan_whose_output_is_deleted_is_looked_at_again(tmp_path, monkeypatch):
    """Codex N-0243a1bb F3: a covered scan stayed covered for the Run on its images alone, even with its output
    gone. Its quiet polls read no coverage and make a bounded number of cheap stats."""
    _raw, executor, identity, covered, shown, stats = _covered_run(tmp_path, monkeypatch, 8901)
    output = tmp_path / 'processed' / 'done_int1d.nexus'
    events = []
    try:
        _wait(executor, events, lambda values: shown or any(event.kind in _TERMINAL for event in values))
        quiet_from = len(stats)
        _settle(executor, events, 2.)
        quiet = (list(covered), len(stats) - quiet_from)
        output.unlink()  # the output goes; its images stay
        _wait(executor, events, lambda values: 2 in _labels(values, 'done') or any(
            event.kind in _TERMINAL for event in values), timeout=20.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert quiet[0] == [('done_int1d.nexus', True)] and quiet[1] <= 5, quiet
    assert sorted(set(_labels(events, 'done'))) == [1, 2] and output.exists()


def test_f3_a_finished_scan_whose_output_is_replaced_is_covered_again_without_showing_it_again(tmp_path, monkeypatch):
    """Codex N-0243a1bb F3: a replaced output is read again for coverage; when it still covers the scan, nothing is
    shown, admitted or written again."""
    _raw, executor, identity, covered, shown, _stats = _covered_run(tmp_path, monkeypatch, 8911)
    output = tmp_path / 'processed' / 'done_int1d.nexus'
    events = []
    try:
        _wait(executor, events, lambda values: shown or any(event.kind in _TERMINAL for event in values))
        copy = output.with_suffix('.copy')
        copy.write_bytes(output.read_bytes())
        os.replace(copy, output)  # same records, another file
        _wait(executor, events, lambda values: len(covered) >= 2 or any(event.kind in _TERMINAL for event in values),
              timeout=20.)
        _settle(executor, events, 1.)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert covered == [('done_int1d.nexus', True)] * 2 and shown == ['done_int1d.nexus'], (covered, shown)
    assert not _labels(events, 'done')
