"""The first resumed Live epoch uses the existing file, just as later epochs do."""
import os

from tests.core.test_h23_c3_composition_append import _whole_output_costs
from tests.xdart.scattering.test_live_owner_transitions import _intent, _saved
from tests.xdart.scattering.test_live_qualify_refusal_run import (
    _frame, _labels, _output, _stop, _terminal, _wait,
)
from tests.xdart.scattering.test_p1b_output_graph import _start
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor


def test_first_live_append_epoch_keeps_prior_rows_without_backup_copy_or_hash(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    _saved(tmp_path, raw, "scan", 2, 125101)
    target = tmp_path / "processed" / "scan_int1d.nexus"
    before = os.stat(target)
    costs = _whole_output_costs(monkeypatch)  # transparent spies on actual transaction operations
    flushes, real_fsync = [], os.fsync

    def fsync(descriptor):
        flushes.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)
    _frame(raw, "scan", 3, "txt", 3.0)
    executor = StandardRunExecutor()
    identity = _start(executor, _intent(tmp_path, raw, live=True), request_value=125102)
    events = []
    try:
        _wait(executor, events, lambda values: 3 in _labels(values) or _terminal(values))
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert _output(tmp_path)[:2] == ((1, 2, 3), (1.0, 2.0, 3.0))
    assert costs == {"stage": 0, "copy": 0, "hash": 0}
    assert flushes == []  # dynamic Live sinks default to buffered writes
    assert os.stat(target).st_ino == before.st_ino
