"""An energy scan in q is reduced on one common grid, from its lowest to its highest energy; new frames that widen it
cannot join the stack its rows were reduced on (maintainer, 2026-10-25, his pick "Rebuild it whole": "A directory Run
reduces the scan again on the wider grid, as a changed scan is rebuilt today. A Live Run stops growing it, names why,
and the next Run rebuilds it"). Before, a directory Append failed the whole Run at the writer. A Live Run keeps
growing the output it writes on its first grid (his later words, 2026-10-25: "it is fine to drop the points that don't
lie on the initial grid rather than stopping"; "It's fine to stick with the energy of the first point if the scan
parameters are not known"); only an earlier Run's output it would extend past its span stops. Real Runs over real
TIFF + .txt scans."""
from __future__ import annotations

import logging
from dataclasses import replace

import fabio
import h5py
import numpy as np
import pytest

from tests.xdart.scattering.test_live_already_processed import _intent
import xrd_tools.reduction.core as core
from tests.xdart.scattering.test_live_qualify_refusal_run import _labels, _said, _settle, _stop, _wait
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal, _start
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor


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


_STOP = "grow_int1d.nexus stops growing: its new frames' energies widen the q grid"
_HC = 1.23984198e-6  # eV m


def _live(tmp_path, raw, request_value, mode="Append"):
    executor = StandardRunExecutor(join_timeout=2.)
    return executor, _start(executor, replace(_intent(tmp_path, raw, live=True), output_mode=mode),
                            request_value=request_value)


def _fresh(tmp_path, energies, request_value):
    """The stack a directory Run over these frames alone writes."""
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    for index, energy in enumerate(energies, 1):
        _frame(raw, "grow", index, energy)
    executor, identity, events = _run_to_terminal(_intent(tmp_path, raw, live=False), request_value=request_value)
    executor.close(identity)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    return _stack(tmp_path)


@pytest.fixture
def reduced_at(monkeypatch):
    """The energy (eV) each frame was reduced at, by label."""
    seen, real = {}, core._ReductionIntegratorProvider.at_frame_energy

    def at_frame_energy(self, integrator, frame, shape, *, warm):
        real(self, integrator, frame, shape, warm=warm)
        seen[int(frame.index)] = round(_HC / float(integrator.wavelength), 3)
    monkeypatch.setattr(core._ReductionIntegratorProvider, "at_frame_energy", at_frame_energy)
    return seen


@pytest.mark.parametrize("mode", ["Append", "Overwrite"])
@pytest.mark.parametrize("energy", [16.3, 16.1])
def test_a_live_energy_scan_grows_on_its_first_grid_and_the_run_goes_on(tmp_path, caplog, reduced_at, mode, energy):
    """Past its span (16.3 keV) or inside it, a frame joins the stack on the grid its first frames were reduced on, at
    the first point's energy (its session lists only theirs), and the Run goes on to the next scan."""
    caplog.set_level(logging.INFO)
    raw = tmp_path / "raw"
    raw.mkdir()
    executor, identity = _live(tmp_path, raw, 9851, mode)
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
    energies = dict(reduced_at)
    labels, axis = _stack(tmp_path)
    assert labels == (1, 2, 3, 4), labels
    assert not any(_STOP in record.getMessage() for record in caplog.records)
    assert {label: energies[label] for label in labels} == {1: 16000., 2: 16100., 3: 16200., 4: 16000.}, energies
    assert np.array_equal(axis, _fresh(tmp_path / "first", [16.0, 16.1, 16.2], 9852)[1])
    assert _stack(tmp_path, "next")[0] == (1,)


@pytest.mark.parametrize("mode", ["Append", "Overwrite"])
def test_a_live_energy_scan_acquired_frame_by_frame_grows_at_its_first_points_energy(tmp_path, reduced_at, mode):
    """The usual case under Live: its first frame is seen alone (one energy), so every later one is reduced at that
    energy, on that frame's grid."""
    raw = tmp_path / "raw"
    raw.mkdir()
    executor, identity = _live(tmp_path, raw, 9861, mode)
    events = []
    try:
        for index, value in enumerate([16.0, 16.1, 16.2], 1):
            _frame(raw, "grow", index, value)
            _wait(executor, events, lambda values, label=index: label in _labels(values, "grow") or _said(
                values, "grow_int1d.nexus stopped growing") or any(event.kind in _TERMINAL for event in values), 20.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    energies = dict(reduced_at)
    labels, axis = _stack(tmp_path)
    assert labels == (1, 2, 3) and {label: energies[label] for label in labels} == {1: 16000., 2: 16000., 3: 16000.}, (
        labels, energies)
    assert np.array_equal(axis, _fresh(tmp_path / "first", [16.0], 9862)[1])


def test_a_live_append_over_an_earlier_runs_energy_scan_past_its_span_stops_it(tmp_path, caplog):
    """An output an earlier Run wrote cannot take rows on another grid: a Live Append stops it, named, and goes on;
    a directory Run reduces it again whole (test_a_directory_append_that_widens_...)."""
    caplog.set_level(logging.INFO)
    raw = tmp_path / "raw"
    raw.mkdir()
    for index, value in enumerate([16.0, 16.1, 16.2], 1):
        _frame(raw, "grow", index, value)
    executor, identity, events = _run_to_terminal(_intent(tmp_path, raw, live=False), request_value=9871)
    executor.close(identity)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    _frame(raw, "grow", 4, 16.3)
    executor, identity = _live(tmp_path, raw, 9872)
    events = []
    try:
        _wait(executor, events, lambda values: _said(values, "grow_int1d.nexus stopped growing") or any(
            event.kind in _TERMINAL for event in values), 20.)
        _frame(raw, "next", 1, 16.0)
        _wait(executor, events, lambda values: 1 in _labels(values, "next") or any(
            event.kind in _TERMINAL for event in values), 20.)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert _stack(tmp_path)[0] == (1, 2, 3)
    assert any(_STOP in record.getMessage() for record in caplog.records)
    assert _stack(tmp_path, "next")[0] == (1,)
