"""An Append onto an output written before a metadata column its new frames carry skips that scan; the Run goes on
(beamline 2026-10-26: a Live Append failed the whole Run on RR_ZrO2_pos3_int2d.nexus, "dirty metadata adds columns
absent on disk: ['epoch_local']", an output written before epoch_local existed; the maintainer's pick "Skip that
scan"). The writer's check is unchanged and typed (MetadataColumnsAbsent). Real Runs over real TIFF + .txt files; the
older output is one this build wrote, its epoch_local column removed."""
from __future__ import annotations

import logging

import h5py

from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _labels, _output, _said, _settle, _stop, _wait
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal, _start
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec

REASON = ("its saved metadata has no epoch_local column, which its new frames carry (an output written by an earlier "
          "xdart); Overwrite reprocesses it")


def _intent(tmp_path, *, live):
    raw, poni = tmp_path / "raw", tmp_path / "cal.poni"
    if not poni.exists():
        write_poni(poni)
    return RunIntent(source_spec=DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format="txt"),
                     poni_file=str(poni), project_root=str(raw), save_path=str(tmp_path / "processed"),
                     output_mode="Append", processing_mode="Int 1D", live_mode=live, max_cores=1,
                     bai_1d_args={"npt": 8, "method": "numpy"})


def _older(tmp_path):
    """Scans "old" and "done", one frame each, written by a first Run; "old" then loses its epoch_local column."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for stem in ("old", "done"):
        _frame(raw, stem, 1, "txt", 1.0)
    executor, identity, events = _run_to_terminal(_intent(tmp_path, live=False), request_value=7741)
    assert next(event for event in events if event.kind in _TERMINAL).kind is StandardEventKind.FINISHED
    executor.close(identity)
    with h5py.File(tmp_path / "processed" / "old_int1d.nexus", "r+") as handle:
        del handle["entry/scan_data/epoch_local"]
    return raw


def test_a_live_append_skips_a_scan_whose_output_predates_its_metadata_columns_and_goes_on(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    raw = _older(tmp_path)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, live=True), request_value=7742)
    events = []
    try:
        _settle(executor, events, 1.)
        _frame(raw, "old", 2, "txt", 2.0)  # the older output's next frame
        _wait(executor, events, lambda values: _said(values, "old_int1d.nexus stopped growing")
              or any(event.kind in _TERMINAL for event in values), 30.)
        _frame(raw, "fresh", 1, "txt", 3.0)  # the Run goes on
        _wait(executor, events, lambda values: 1 in _labels(values, "fresh")
              or any(event.kind in _TERMINAL for event in values), 30.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert any(message.startswith("old_int1d.nexus stops growing") and REASON in message
               for message in (record.getMessage() for record in caplog.records))
    assert _output(tmp_path, "old")[:2] == ((1,), (1.0,))  # its saved frame kept as it was
    assert _output(tmp_path, "fresh")[:2] == ((1,), (3.0,))


def test_a_directory_append_skips_a_scan_whose_output_predates_its_metadata_columns_and_finishes(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    raw = _older(tmp_path)
    for stem in ("old", "done"):
        _frame(raw, stem, 2, "txt", 2.0)
    executor, identity, events = _run_to_terminal(_intent(tmp_path, live=False), request_value=7743)
    terminal = next(event for event in events if event.kind in _TERMINAL)
    executor.close(identity)
    assert terminal.kind is StandardEventKind.FINISHED, terminal.detail
    assert any(message.startswith("old_int1d.nexus skipped") and REASON in message
               for message in (record.getMessage() for record in caplog.records))
    assert _output(tmp_path, "old")[:2] == ((1,), (1.0,))
    assert _output(tmp_path, "done")[:2] == ((1, 2), (1.0, 2.0))
