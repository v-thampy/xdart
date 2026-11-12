"""B-E, the frozen oracle of NEXT-1210-VERIFY-ONCE: equal science through every writer path.

The same frames reduced through a finite Overwrite, a fresh folder Append and a Live Run, each output reloaded from
disk: the saved 1D and 2D science, uncertainties, axes, frame labels and metadata are equal, exactly. No test compared
these three since the live=batch=reload spine test went with the Static Scan page (6c701391); the verify-once changes
move each path's write verification, so this row must pass on the parent and after every commit of that objective.
Real data (the Combi4 scan in XDART_TEST_DATA) when present; a synthetic twin always (CI has no test data).
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.xdart.scattering.test_live_owner_transitions import _intent
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _stop, _terminals
from tests.xdart.scattering.test_p1b_output_graph import _run_to_terminal, _start
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xrd_tools.core.frame_view import assert_frameview_equivalent
from xrd_tools.io.frame_view import read_frame_view

_COMBI4 = "Combi4_Angledependence_samz_4p9_03271005"
_COMBI4_PONI = "LaB6_detz190_dety72_th5_03261554_0001.poni"


def _source(tmp_path, data):
    """The raw folder, its scan stem, frame count and PONI (None: the folder intent writes its own)."""
    raw = tmp_path / "raw"
    raw.mkdir()
    if data == "synthetic":
        for index in range(1, 7):
            _frame(raw, "scan", index, "txt", float(index))
        return raw, "scan", 6, None
    root = os.environ.get("XDART_TEST_DATA")
    tiff = Path(root or "XDART_TEST_DATA-is-unset") / "Tiff"
    members = sorted(tiff.glob(f"{_COMBI4}_*.tif"))
    if len(members) != 16 or not (tiff / _COMBI4_PONI).is_file():
        pytest.skip("XDART_TEST_DATA with the 16-frame Combi4 scan and its PONI is needed")
    for member in members:
        shutil.copy2(member, raw / member.name)
        shutil.copy2(member.with_suffix(".txt"), raw / member.with_suffix(".txt").name)
    return raw, _COMBI4, 16, tiff / _COMBI4_PONI


def _ready(events, stem):
    return {event.frame_key.local_frame_label for event in events
            if event.kind is StandardEventKind.FRAME_READY and Path(str(event.artifact)).name.startswith(stem)}


def _reduce(tmp_path, raw, stem, count, poni, kind):
    intent = _intent(tmp_path, raw, live=kind == "live", mode="Int 2D",
                     output="Overwrite" if kind == "overwrite" else "Append")
    if poni is not None:
        intent.poni_file = str(poni)
    intent.save_path = str(tmp_path / kind)
    if kind == "live":
        executor = StandardRunExecutor(join_timeout=5.0)
        identity, events = _start(executor, intent, request_value=11), []
        deadline = time.monotonic() + 180.0
        while len(_ready(events, stem)) < count and time.monotonic() < deadline:
            events.extend(executor.drain_events())
            time.sleep(0.01)
        terminals = _stop(executor, identity, events)
    else:
        executor, identity, events = _run_to_terminal(intent, request_value=11)
        executor.close(identity)
        terminals = _terminals(events)
    outputs = sorted((tmp_path / kind).rglob("*.nexus"))
    assert len(outputs) == 1, (kind, outputs, terminals)
    return outputs[0], terminals


def _labels(path):
    with h5py.File(path, "r") as handle:
        return {dimension: handle[f"entry/{dimension}/frame_index"][()].tolist()
                for dimension in ("integrated_1d", "integrated_2d")}


def _exactly_equal(name, a, b):
    if a is None or b is None:
        assert a is None and b is None, f"{name}: one side is None"
        return
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape, (name, a.dtype, b.dtype, a.shape, b.shape)
    np.testing.assert_array_equal(a, b, err_msg=name)


@pytest.mark.parametrize("data", ["synthetic", "combi4"])
def test_finite_overwrite_fresh_append_and_live_save_equal_science(tmp_path, data):
    raw, stem, count, poni = _source(tmp_path, data)
    saved = {kind: _reduce(tmp_path, raw, stem, count, poni, kind) for kind in ("overwrite", "append", "live")}
    labels = {kind: _labels(path) for kind, (path, _terminal) in saved.items()}
    expected = list(range(1, count + 1))
    assert labels["overwrite"] == {"integrated_1d": expected, "integrated_2d": expected}, labels
    assert labels["append"] == labels["overwrite"] and labels["live"] == labels["overwrite"], labels
    reference = saved["overwrite"][0]
    for label in expected:
        base = read_frame_view(reference, label)
        assert base.intensity_1d is not None and base.intensity_2d is not None
        for kind in ("append", "live"):
            other = read_frame_view(saved[kind][0], label)
            where = f"{kind} frame {label}"
            for field in ("intensity_1d", "sigma_1d", "intensity_2d"):
                _exactly_equal(f"{where} {field}", getattr(base, field), getattr(other, field))
            for axis in ("axis_1d", "axis_2d_x", "axis_2d_y"):
                a, b = getattr(base, axis), getattr(other, axis)
                assert (a.label, a.unit) == (b.label, b.unit), (where, axis, a, b)
                _exactly_equal(f"{where} {axis}", a.values, b.values)
            assert dict(base.metadata_numeric) == dict(other.metadata_numeric), where
            assert dict(base.metadata_raw) == dict(other.metadata_raw), where
            assert_frameview_equivalent(base, other)
