"""A Live directory Append leaves a scan it has settled alone until its images change (maintainer,
2026-10-23: "If something has already been processed, it should be skipped anyway"; Codex,
DESIGN-append-image-reprocess-125: a watch, never a permanent skip, and never for the output a Live epoch owns).

Beamline 2026-10-23 (dev 21f4872d, Windows share): a Live Run read every finished scan's sidecars again before its
first frame (over 40 s), then refused finished scans one by one after the beamline touched their files, and,
settling the scan still growing on the way, failed it at its next frame: "cannot bind an allocation to a populated
store"."""
from __future__ import annotations

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
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    stopped = next(event for event in events if event.kind is StandardEventKind.STOPPED)
    assert stopped.warnings == () and not _said(events, 'stopped growing'), (stopped.warnings, _terminals(events))
    assert 'done_int1d.nexus' not in admitted[before_touch:], admitted
    assert sorted(set(_labels(events, 'grow'))) == [1, 2]
