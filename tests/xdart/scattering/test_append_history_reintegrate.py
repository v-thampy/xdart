"""C-12 of NEXT-1210-RESPONSIVENESS: Reintegrate reads a version-2 Append history.

An image series reduced, grown and appended in a second Run has a two-epoch history, which C-12 saves as version 2
(each epoch its new images). A full Reintegrate from that history, with the images moved along with their Project (a
relocated source root), commits and reproduces the saved curves, as it does from a version-1 history.
"""
from __future__ import annotations

import h5py
import numpy as np

from tests.xdart.scattering._output_slots import written
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _intent, _run_to_terminal, _write_tiff, write_poni


def _finished(intent, request_value):
    executor, identity, events = _run_to_terminal(intent, request_value=request_value)
    try:
        assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"], events
    finally:
        executor.close(identity)


def test_a_two_epoch_append_history_reintegrates_from_a_moved_root(tmp_path, monkeypatch):
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xrd_tools.io import append, open_scan
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.reduction import ReintegrateSuccessorPlan, run_reintegrate_successor
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    raw = tmp_path / "raw"
    raw.mkdir()
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    for label in (1, 2, 3):
        _write_tiff(raw / f"scan_{label:04d}.tif", label)
    target = tmp_path / "scan.nexus"
    _finished(_intent(raw / "scan_0001.tif", target, poni, output_mode="Append"), 9501)
    for label in (4, 5):
        _write_tiff(raw / f"scan_{label:04d}.tif", label)
    _finished(_intent(raw / "scan_0001.tif", target, poni, output_mode="Append"), 9502)
    artifact = written(target, "Int 1D")
    with h5py.File(artifact, "r") as handle:
        lineage = append._read_lineage(handle["entry"])
    assert lineage["version"] == 2 and len(lineage["epochs"]) == 2, (lineage["version"], len(lineage["epochs"]))
    assert [len(epoch["source"]["image_members"]) for epoch in lineage["epochs"]] == [3, 2]
    moved = tmp_path / "moved"
    raw.rename(moved)
    predecessor = open_scan(artifact)
    before = predecessor.get_1d(None)
    intent = _intent(moved / "scan_0001.tif", target, poni, output_mode="Append")
    plan = ReintegrateSuccessorPlan.from_artifact(
        artifact, entry="entry", dimension="1d", source_root=str(moved),
        expected_target_snapshot=capture_target_snapshot(artifact),
        expected_labels=tuple(int(label) for label in predecessor.frame_indices),
        preparation=ScatteringWorkspace._reintegrate_preparation(intent, "1d"))
    result = run_reintegrate_successor(plan)
    assert result.disposition == "COMMITTED", result
    after = open_scan(result.output_artifact).get_1d(None)
    np.testing.assert_array_equal(after.q, before.q)
    np.testing.assert_array_equal(after.intensity, before.intensity)
