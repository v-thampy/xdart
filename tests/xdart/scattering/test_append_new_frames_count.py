"""An Append Run says how many of its processed frames are new (maintainer, 2026-10-25: "at the end of an append run
append the new frames processed in brackets. right now it just says something like 35 processed. it will be good to
know how many of these were newly processed immediately in the readiness status as well").

Real directory Runs over real TIFF + .txt scans: a first Append, the scan grown by one frame, then nothing new; and a
Replace, which counts no "new"."""
from __future__ import annotations

import logging
from dataclasses import replace

from tests.xdart.scattering.test_live_already_processed import _intent
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal
from xdart.gui.tabs.scattering.page import _directory_file_progress


def _terminal(intent, request_value):
    executor, identity, events = _run_to_terminal(intent, request_value=request_value)
    executor.close(identity)
    terminals = [event for event in events if event.kind in _TERMINAL]
    assert [event.kind.name for event in terminals] == ["FINISHED"], terminals
    return terminals[0]


def _end_of_run(caplog) -> str:
    return next(record.getMessage() for record in reversed(caplog.records) if "End of Run" in record.getMessage())


def test_an_append_run_counts_the_frames_it_saved_apart_from_those_its_outputs_held(tmp_path, caplog):
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in (1, 2, 3):
        _frame(raw, "grow", index, "txt", float(index))
    intent = _intent(tmp_path, raw, live=False)
    caplog.set_level(logging.INFO)

    first = _terminal(intent, 9901)
    assert (first.files_processed, first.frames_new) == (3, 3)
    assert "3 files processed (3 new), 0 skipped" in _end_of_run(caplog)
    assert _directory_file_progress(first).text("Complete") == "Complete · 3/3 processed (3 new)"

    _frame(raw, "grow", 4, "txt", 4.0)
    grown = _terminal(intent, 9902)
    assert (grown.files_processed, grown.frames_new) == (4, 1)
    assert "4 files processed (1 new), 0 skipped" in _end_of_run(caplog)

    again = _terminal(intent, 9903)
    assert (again.files_processed, again.frames_new) == (4, 0)
    assert "4 files processed (0 new), 0 skipped" in _end_of_run(caplog)

    replaced = _terminal(replace(intent, output_mode="Overwrite"), 9904)
    assert (replaced.files_processed, replaced.frames_new) == (4, None)
    assert "4 files processed, 0 skipped" in _end_of_run(caplog)
