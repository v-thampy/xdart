"""The Run status counts a scan's files as its frames are processed (maintainer, 2026-10-26: "if a scan with 10 frames
already exists on disk, the status shows (0/10) processed till all 10 are processed. but as new frames come in for a
scan and it's all caught up, then it says (10/11) while processing and (11/11)"). Buffered outputs are durable at
their fold's end, so the durable count alone held at 0 through a scan already on disk. The status counts the shown
frames (display only); each event's files_processed stays the durable count ("FRAME_READY is display/reduction truth,
not a durable file receipt", test_p1b_b14). The real executor, the page's progress builder and Live Runs over real TIFF
+ .txt files."""
from __future__ import annotations

import pytest

pytest.importorskip("pyqtgraph")

from tests.xdart.scattering.test_browse_reference_rings_run import _poni
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _settle, _stop, _wait
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _start
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.page import _directory_file_progress
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


def _intent(root, mode):
    raw = root / "raw"
    return RunIntent(source_spec=DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format="txt"),
                     poni_file=str(root / "cal.poni"), project_root=str(raw), save_path=str(root / "processed"),
                     output_mode=mode, processing_mode="Int 1D", live_mode=True, max_cores=1,
                     bai_1d_args={"npt": 8, "method": "numpy"})


def _counts(events):
    """What the status shows at each FRAME_READY: (processed, discovered, new), and the event's durable count."""
    shown = []
    for event in events:
        if event.kind is StandardEventKind.FRAME_READY:
            progress = _directory_file_progress(event)
            shown.append((progress.processed, progress.discovered, progress.new, event.files_processed))
    return shown


def _run(root, mode, request_value, grow=None):
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(root, mode), request_value=request_value)
    events = []
    try:
        if grow is not None:
            _settle(executor, events, 1.)
            grow()
        _wait(executor, events, lambda values: sum(e.kind is StandardEventKind.FRAME_READY for e in values) >= (
            1 if grow is not None else 5) or any(e.kind in _TERMINAL for e in values), 60.)
        _settle(executor, events, 1.)
    finally:
        _stop(executor, identity, events)
    return _counts(events)


@pytest.fixture
def scan_on_disk(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _poni(tmp_path)
    for index in range(1, 6):
        _frame(raw, "scan", index, "txt", float(index))
    return raw


def test_a_scan_already_on_disk_counts_up_frame_by_frame(tmp_path, scan_on_disk):
    counts = _run(tmp_path, "Overwrite", 7751)
    processed = [row[0] for row in counts]
    assert processed and processed[-1] == 5, counts
    assert processed[0] < 5 and processed == sorted(processed), counts  # it counted as it went, never backwards
    assert all(row[1] == 5 for row in counts), counts
    assert counts[0][3] == 0, counts  # the durable count is unchanged: nothing of the buffered scan is saved yet


def test_an_append_onto_a_grown_scan_never_counts_its_saved_frames_twice(tmp_path, scan_on_disk):
    _run(tmp_path, "Overwrite", 7752)  # the scan's output, all five frames
    counts = _run(tmp_path, "Append", 7753, grow=lambda: _frame(scan_on_disk, "scan", 6, "txt", 6.0))
    assert counts, "the grown frame was never shown"
    assert all(row[0] <= row[1] == 6 for row in counts), counts
    assert counts[-1][0] == 6 and counts[-1][2] == 1, counts  # all six processed, one of them new


def test_a_reopened_prefix_does_not_count_against_other_pending_files(tmp_path, scan_on_disk):
    _run(tmp_path, "Overwrite", 7754)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, "Append"), request_value=7755)
    events = []
    try:
        _wait(executor, events, lambda values: any(e.files_processed == 5 for e in values), 30.)
        _frame(scan_on_disk, "scan", 6, "txt", 6.0)
        _frame(scan_on_disk, "waiting", 1, "txt", 1.0, sidecar=False)
        _wait(executor, events, lambda values: bool(_counts(values)) or any(e.kind in _TERMINAL for e in values), 30.)
        before = [e for e in events if e.kind is StandardEventKind.DISCOVERY
                  and e.artifact.endswith("scan_int1d.nexus") and e.artifact_total == 6
                  and e.artifact_completed == 5]
        assert before, "the reopened scan was not announced before its tail was reduced"
        assert all(_directory_file_progress(e).processed == 5 for e in before), [
            (e.files_processed, e.files_shown, e.files_pending) for e in before]
        assert all(row[0] == 6 and row[2] == 1 for row in _counts(events)), _counts(events)
    finally:
        _stop(executor, identity, events)
