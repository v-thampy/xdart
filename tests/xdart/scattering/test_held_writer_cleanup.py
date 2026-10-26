"""A Live Append whose in-place reopen fails its checkpoint seal still ends clean, so Run unlocks and xdart can close
(beamline 2026-10-26, Windows share: "writer flush incomplete: stream-checkpoint descriptor/path identity changed during
seal: ... named=(..., 1793048641012136700, 0) descriptor=(..., 1793048674507614200, ...)", the pathname view 33 s behind
the open descriptor; then "[RUN-CLEANUP] ... failures=('dynamic_output.finish: writer flush incomplete: checkpoint
requires EXECUTING, got integrity-hold',)" at every retry, the status "Standard cleanup remains pending", Run disabled,
and closing xdart retried forever until it was killed). The real executor, page and Live Runs over real TIFF + .txt
files; the seal's refusal is the transaction's own TargetChanged, raised once, as the share's stale view raised it."""
from __future__ import annotations

import pytest

pytest.importorskip("pyqtgraph")

from tests.xdart.scattering.test_browse_reference_rings_run import _page, _poni, _pump
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _settle, _stop, _wait
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _start
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus
from xrd_tools.io import output_transaction
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


def _intent(root):
    raw = root / "raw"
    return RunIntent(source_spec=DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format="txt"),
                     poni_file=str(root / "cal.poni"), project_root=str(raw), save_path=str(root / "processed"),
                     output_mode="Append", processing_mode="Int 1D", live_mode=True, max_cores=1,
                     bai_1d_args={"npt": 8, "method": "numpy"})


def _frames(events):
    return sum(1 for event in events if event.kind is StandardEventKind.FRAME_READY)


@pytest.fixture
def written(tmp_path):
    """One scan's output with one frame, written by a first Live Run; the folder of its images."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _poni(tmp_path)
    _frame(raw, "old", 1, "txt", 1.0)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path), request_value=7741)
    events = []
    _wait(executor, events, lambda values: _frames(values) >= 1 or any(e.kind in _TERMINAL for e in values))
    _settle(executor, events, .5)
    _stop(executor, identity, events)
    assert (tmp_path / "processed" / "old_int1d.nexus").is_file()
    return raw


@pytest.fixture
def stale_seal(monkeypatch):
    """The next stream-checkpoint seal refuses as the share's stale pathname view made it refuse; later seals pass."""
    seal, refused = output_transaction._descriptor_stream_stat_receipt, []

    def once(descriptor, path, **kwargs):
        if kwargs.get("role") == "stream-checkpoint" and not refused:
            refused.append(str(path))
            raise output_transaction.TargetChanged(
                f"stream-checkpoint descriptor/path identity changed during seal: {path} (a stale pathname view)")
        return seal(descriptor, path, **kwargs)
    monkeypatch.setattr(output_transaction, "_descriptor_stream_stat_receipt", once)
    return refused


def test_a_live_append_whose_reopened_output_fails_its_seal_ends_clean(tmp_path, written, stale_seal):
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path), request_value=7742)
    events = []
    try:
        _settle(executor, events, 1.)
        _frame(written, "old", 2, "txt", 2.0)  # the scan grows: its output is reopened in place
        _wait(executor, events, lambda values: any(e.kind in _TERMINAL for e in values) or bool(stale_seal), 30.)
        _settle(executor, events, 1.)
    finally:
        terminal = _stop(executor, identity, events)
    assert stale_seal, "the reopened output was never sealed"
    run = executor._exact_run(identity)
    failures = () if run is None else tuple(f"{e.operation}: {e.message}" for e in run.cleanup_failures)
    retained = () if run is None else tuple(name for name in ("session", "sink", "output", "source", "resources")
                                            if getattr(run, name) is not None)
    ended = [kind for kind, _detail in terminal]
    assert ended and ended[-1] in {"FAILED", "STOPPED"}, terminal
    receipt = executor.close(identity)
    assert receipt.cleanup_status is CleanupStatus.CLEANED, (receipt.cleanup_status, retained, failures)


def test_after_that_failure_the_page_unlocks_run(qapp, tmp_path, written, stale_seal):  # noqa: F811
    with _page(qapp, _intent(tmp_path)) as (page, _view, observed):
        page._shell.run_controls.startButton.click()
        assert _pump(qapp, lambda: page._lifecycle.active_run_identity is not None, 30.)
        _pump(qapp, lambda: False, 1.)
        _frame(written, "old", 2, "txt", 2.0)
        assert _pump(qapp, lambda: bool(stale_seal), 30.), "the reopened output was never sealed"
        ended = _pump(qapp, lambda: any(event.kind in _TERMINAL for event in observed), 30.)
        if not ended:  # the scan was skipped and the Run goes on: stop it
            page._shell.run_controls.stopButton.click()
            assert _pump(qapp, lambda: any(event.kind in _TERMINAL for event in observed), 30.)
        assert _pump(qapp, lambda: page._start_permitted() == (True, ""), 15.), (
            page._start_permitted(), page._shell.run_controls.readinessLabel.full_text())


def test_the_held_output_is_left_as_an_unfinished_append_that_names_replace(tmp_path, written, stale_seal, caplog):
    """The output the failed seal held is left as any interrupted in-place Append leaves it (1.2.5 notes: "The next
    Append skips it and says to use Replace"); the beamline's next Run said so for LaB6_..._10261403_int2d.nexus."""
    import logging

    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path), request_value=7744)
    events = []
    try:
        _settle(executor, events, 1.)
        _frame(written, "old", 2, "txt", 2.0)
        _wait(executor, events, lambda values: any(e.kind in _TERMINAL for e in values) or bool(stale_seal), 30.)
        _settle(executor, events, 1.)
    finally:
        _stop(executor, identity, events)
    assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED
    caplog.set_level(logging.INFO)
    again = StandardRunExecutor(join_timeout=2.)
    second = _start(again, _intent(tmp_path), request_value=7745)
    later = []
    try:
        _frame(written, "old", 3, "txt", 3.0)
        _wait(again, later, lambda values: any("use Replace" in record.getMessage() for record in caplog.records)
              or any(e.kind in _TERMINAL for e in values), 30.)
    finally:
        _stop(again, second, later)
    assert any("old_int1d.nexus" in record.getMessage() and "use Replace" in record.getMessage()
               for record in caplog.records), [record.getMessage() for record in caplog.records][-8:]
