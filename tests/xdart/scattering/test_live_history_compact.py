"""C-12 of NEXT-1210-RESPONSIVENESS on a real Live Run: its saved Append history stores each image once.

The maintainer, 2026-11-12 (verbatim): "Add it to packet C now"; "Save only each epoch's new images (Recommended)".
A Live Int 1D Run whose frames arrive one at a time commits one epoch per frame. Its history is version 2 and stores
each image once; its committed prefix decodes; Reintegrate admits every saved frame from it; and a later finite Append
of the same folder extends it.
"""
from __future__ import annotations

import h5py

from tests.xdart.scattering.test_live_owner_transitions import _intent
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _labels, _settle, _stop, _wait
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal, _start
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xrd_tools.io import append
from xrd_tools.io.record_writer import _decode_replacement_fact
from xrd_tools.reduction import reintegrate


def _history(path):
    with h5py.File(path, "r") as handle:
        lineage = append._read_lineage(handle["entry"])
        prefix = append.decode_committed_append_prefix(handle)
    return lineage, prefix


def test_a_live_runs_history_stores_each_image_once_and_stays_usable(tmp_path, monkeypatch):
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    raw = tmp_path / "raw"
    raw.mkdir()
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, live=True), request_value=9401)
    events = []
    try:
        for index in range(1, 5):  # one frame per Live poll: one epoch each
            _frame(raw, "grow", index, "txt", float(index))
            _wait(executor, events, lambda values, i=index: i in _labels(values, "grow"))
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    output = tmp_path / "processed" / "grow_int1d.nexus"
    lineage, prefix = _history(output)
    assert lineage["version"] == 2 and len(lineage["epochs"]) >= 2, (lineage["version"], len(lineage["epochs"]))
    stored = sum(len(epoch["source"]["image_members"]) for epoch in lineage["epochs"])
    assert stored == len(prefix.committed_labels) == 4  # each image once
    assert append._expand_lineage(lineage)["epochs"][-1]["source"] == append._source_dict(prefix.intent.source)
    with h5py.File(output, "r") as handle:
        facts = [_decode_replacement_fact(handle, label) for label in prefix.committed_labels]
    for fact in facts:  # Reintegrate's lineage admission for each saved frame, from the version-2 history
        reintegrate._admit_source_topology(fact, full_inventory=True, selected_labels=(fact["label"],))
    for index in (5, 6):
        _frame(raw, "grow", index, "txt", float(index))
    executor, identity, events = _run_to_terminal(_intent(tmp_path, raw, live=False), request_value=9402)
    executor.close(identity)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    lineage, prefix = _history(output)
    assert lineage["version"] == 2 and len(prefix.committed_labels) == 6
    assert sum(len(epoch["source"]["image_members"]) for epoch in lineage["epochs"]) == 6
