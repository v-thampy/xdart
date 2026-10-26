"""Codex N-acb7b192's findings on the beamline batch (2026-10-25), each through its production owner.

A-R1: an output saved while a win32 pathname ctime still went into the Append source digest is recognized, not
refused at every Append. B-R1: the Metadata window says why it is empty before a Run's first frame too. B-R2: a failed
Run whose cleanup is pending keeps the scan list locked, as Browse refuses. B-R3: a one-scan Append says its new frames
while it runs. B-R4: a File Type spelled without its dot keeps the container output policy.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import pytest

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_e1b2_page_command_boundaries import _Executor, _active_page, _dispose
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _intent, _run_to_terminal, _write_tiff
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import FatalExecution
from xrd_tools.io import stat_identity
from xrd_tools.io.append import AppendDisposition, decode_committed_append_prefix, qualify_append
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xrd_tools.sources.execution_graph import append_source_from_execution_graph, qualify_source_execution_graph
from xrd_tools.sources.selection import DirectorySourceSpec, holds_containers


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _single(tmp_path: Path, **options):
    raw, target, poni = tmp_path / "frame_0001.tif", tmp_path / "saved.nexus", tmp_path / "geometry.poni"
    _write_tiff(raw, 3)
    write_poni(poni)
    return _intent(raw, target, poni, **options)


def _finish(intent, request_value):
    executor, identity, events = _run_to_terminal(intent, request_value=request_value)
    executor.close(identity)
    terminals = [event for event in events if event.kind in _TERMINAL]
    assert [event.kind.name for event in terminals] == ["FINISHED"], terminals
    return terminals[0], events


def test_a_r1_an_output_saved_before_ctime_was_settled_is_recognized_at_every_append(
        tmp_path, monkeypatch) -> None:
    # On POSIX an exact ctime slot is the old win32 rule: the output records the raw-ctime digest.
    monkeypatch.setattr(stat_identity, "IDENTITY_CARRIES_CTIME", True)
    intent = _single(tmp_path, output_mode="Append")
    terminal, _events = _finish(intent, 10201)
    output = Path(terminal.artifact)
    with h5py.File(output, "r") as handle:
        prior = decode_committed_append_prefix(handle, entry="entry")
    before = output.read_bytes()
    monkeypatch.setattr(stat_identity, "IDENTITY_CARRIES_CTIME", False)  # the settled rule now
    source = append_source_from_execution_graph(qualify_source_execution_graph(intent.source_spec), generation=2)
    assert source.legacy_digest == prior.intent.source.digest != source.digest
    current = replace(prior.intent, source=source)
    decisions = [qualify_append(output, current), qualify_append(output, current)]
    assert [decision.disposition for decision in decisions] == [AppendDisposition.SKIP] * 2, [
        (decision.disposition.name, decision.reason) for decision in decisions]
    assert output.read_bytes() == before


def test_b_r1_metadata_before_the_first_frame_explains_the_run_hold(qapp) -> None:
    page, lifecycle, _identity = _active_page(_Executor())
    try:
        assert lifecycle.phase.value == "running" and page._context_controller.navigation.current is None
        page._open_analysis_mount("metadata")
        assert page._metadata_dialog.status.text() == (
            "Metadata is available after the Run stops; then press Metadata again.")
    finally:
        _dispose(page, qapp)


def test_b_r2_a_failed_run_with_cleanup_pending_keeps_the_scan_list_locked(qapp) -> None:
    page, lifecycle, identity = _active_page(_Executor())
    try:
        lifecycle.fatal(FatalExecution(identity))
        assert lifecycle.phase.value == "failed" and not lifecycle.reset_permitted
        assert not page._context_controller.browse_lifecycle_admissible()
        page._refresh_shell()
        browser = page._shell.browser
        assert not browser.scans.isEnabled()
        assert browser.scans.toolTip() == "Other scans open once the failed Run has finished cleaning up."
    finally:
        _dispose(page, qapp)


def test_b_r3_a_one_scan_append_says_its_new_frames_while_it_runs(tmp_path) -> None:
    terminal, events = _finish(_single(tmp_path, output_mode="Append"), 10202)
    assert terminal.frames_new == 1
    # Only saved frames are new: a frame's FRAME_READY comes before its frame is saved, so it reads "(0 new)" here.
    shown = [event.detail for event in events if event.kind is StandardEventKind.FRAME_READY]
    assert shown and all(detail.endswith(" new)") for detail in shown), shown
    (tmp_path / "replace").mkdir()
    replaced, events = _finish(replace(_single(tmp_path / "replace"), output_mode="Overwrite"), 10203)
    assert replaced.frames_new is None
    assert not any(" new)" in event.detail for event in events if event.kind is StandardEventKind.FRAME_READY)


def test_b_r4_a_dotless_container_suffix_keeps_the_container_policy(tmp_path) -> None:
    source = DirectorySourceSpec(tmp_path, recursive=True, suffixes=("nexus",))
    raw = tmp_path / "raw.nexus"
    with h5py.File(raw, "w") as handle:
        handle.create_group("entry")
    session = DirectoryIndexSession(probe_candidates=False)
    try:
        session.configure(source.root, recursive=source.recursive, suffixes=source.suffixes)
        assert raw in [candidate.path for candidate in session.observe(refresh=True).discovered_snapshot.candidates]
    finally:
        session.close()
    assert holds_containers(source.suffixes)
    assert holds_containers(("_master.h5",)) and holds_containers(("h5",))
    assert not holds_containers((".tif",)) and not holds_containers(("cbf",)) and not holds_containers(("raw",))
