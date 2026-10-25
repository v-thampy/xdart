"""An energy scan in q is reduced on one common grid, from its lowest to its highest energy; new frames that widen it
cannot join the stack its rows were reduced on (maintainer, 2026-10-25, his pick "Rebuild it whole": "A directory Run
reduces the scan again on the wider grid, as a changed scan is rebuilt today. A Live Run stops growing it, names why,
and the next Run rebuilds it"). Before, a directory Append failed the whole Run at the writer and a Live Run reduced the
new frames on the old grid. Real Runs over real TIFF + .txt scans."""
from __future__ import annotations

import logging

import fabio
import h5py
import numpy as np
import pytest

from tests.xdart.scattering.test_live_already_processed import _intent
from tests.xdart.scattering.test_live_never_waits import _live
from tests.xdart.scattering.test_live_qualify_refusal_run import _labels, _said, _settle, _stop, _wait
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal


def _frame(raw, stem, index, energy):
    raw.joinpath(f"{stem}_{index:04d}.txt").write_text(
        f"# Counters\nI0 = {index}\n# Motors\nenergy = {energy!r}\nUser: energy, time: Mon Jan 15 10:30:00 2024  # Temp\n")
    partial = raw / f"{stem}_{index:04d}.part"
    fabio.tifimage.TifImage(data=np.full((195, 487), index, dtype=np.uint16)).write(str(partial))
    partial.replace(raw / f"{stem}_{index:04d}.tif")


def _stack(tmp_path, stem="grow"):
    with h5py.File(tmp_path / "processed" / f"{stem}_int1d.nexus", "r") as handle:
        return (tuple(handle["entry/integrated_1d/frame_index"][()].tolist()),
                np.asarray(handle["entry/integrated_1d/axis_1"][()]))


def _directory(tmp_path, before, after):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    for index, energy in enumerate(before, 1):
        _frame(raw, "grow", index, energy)
    executor, identity, events = _run_to_terminal(_intent(tmp_path, raw, live=False), request_value=9841)
    executor.close(identity)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    for index, energy in enumerate(after, len(before) + 1):
        _frame(raw, "grow", index, energy)
    executor, identity, events = _run_to_terminal(_intent(tmp_path, raw, live=False), request_value=9842)
    executor.close(identity)
    return [event for event in events if event.kind in _TERMINAL]


@pytest.mark.parametrize("case, before, after", [
    ("its span widens", [16.0, 16.1, 16.2], [16.3]),
    ("a common energy starts to vary", [16.0, 16.0, 16.0], [16.3]),
])
def test_a_directory_append_that_widens_an_energy_scans_grid_reduces_it_again_whole(tmp_path, caplog, case, before,
                                                                                     after):
    caplog.set_level(logging.INFO)
    terminals = _directory(tmp_path / "append", before, after)
    assert [event.kind.name for event in terminals] == ["FINISHED"], terminals
    assert any("grow_int1d.nexus" in rebuilt for rebuilt in terminals[0].rebuilt), terminals[0].rebuilt
    assert any("widen the q grid its frames were reduced on, so the whole scan is reprocessed" in record.getMessage()
               for record in caplog.records)
    labels, axis = _stack(tmp_path / "append")
    # The rebuilt stack has the grid a first Run over all its frames gives.
    fresh = tmp_path / "fresh" / "raw"
    fresh.mkdir(parents=True)
    for index, energy in enumerate(before + after, 1):
        _frame(fresh, "grow", index, energy)
    executor, identity, events = _run_to_terminal(_intent(tmp_path / "fresh", fresh, live=False), request_value=9843)
    executor.close(identity)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    fresh_labels, fresh_axis = _stack(tmp_path / "fresh")
    assert labels == fresh_labels == tuple(range(1, len(before + after) + 1)), (labels, fresh_labels)
    assert np.array_equal(axis, fresh_axis), (case, axis[[0, -1]], fresh_axis[[0, -1]])


def test_a_directory_append_inside_an_energy_scans_span_extends_it(tmp_path):
    terminals = _directory(tmp_path, [16.0, 16.1, 16.2], [16.1])
    assert [event.kind.name for event in terminals] == ["FINISHED"] and terminals[0].rebuilt == (), terminals
    assert _stack(tmp_path)[0] == (1, 2, 3, 4)


@pytest.mark.parametrize("energy, grows", [(16.3, False), (16.1, True)])
def test_a_live_energy_scans_frame_outside_its_span_stops_it_and_the_run_goes_on(tmp_path, caplog, energy, grows):
    caplog.set_level(logging.INFO)
    raw = tmp_path / "raw"
    raw.mkdir()
    executor, identity = _live(tmp_path, raw, 9851)
    events = []
    try:
        for index, value in enumerate([16.0, 16.1, 16.2], 1):
            _frame(raw, "grow", index, value)
        _wait(executor, events, lambda values: 3 in _labels(values, "grow") or any(
            event.kind in _TERMINAL for event in values), 20.)
        _frame(raw, "grow", 4, energy)
        _wait(executor, events, lambda values: 4 in _labels(values, "grow") or _said(
            values, "grow_int1d.nexus stopped growing") or any(event.kind in _TERMINAL for event in values), 20.)
        _frame(raw, "next", 1, 16.0)
        _wait(executor, events, lambda values: 1 in _labels(values, "next") or any(
            event.kind in _TERMINAL for event in values), 20.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    labels, _axis = _stack(tmp_path)
    assert labels == ((1, 2, 3, 4) if grows else (1, 2, 3)), labels
    assert grows is not any("grow_int1d.nexus stops growing: its new frames' energies widen the q grid" in
                            record.getMessage() for record in caplog.records)
    assert _stack(tmp_path, "next")[0] == (1,)
