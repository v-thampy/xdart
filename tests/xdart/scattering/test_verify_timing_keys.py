"""B-4 of NEXT-1210-VERIFY-ONCE: the Run's timing summary says where verification time went.

With Performance Monitoring on (XDART_PERF), the writer reports its checkpoint readbacks and its close readback
beside the fast science readback, close and terminal, in the terminal timing details and the [PERF-WRITER-FINISH]
line, so the maintainer's mapped-drive before-and-after reads each one.
"""
from __future__ import annotations

import logging
import time

from tests.xdart.scattering.test_live_owner_transitions import _intent
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _start
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_values import StandardEventKind


def _append(tmp_path, raw, save):
    intent = _intent(tmp_path, raw, live=False, output="Append")
    intent.save_path = str(save)
    executor = StandardRunExecutor(join_timeout=5.0)
    identity, events, deadline = _start(executor, intent, request_value=5), [], time.monotonic() + 120
    try:
        while not any(event.kind in _TERMINAL for event in events):
            assert time.monotonic() < deadline, "no terminal"
            events.extend(executor.drain_events())
            time.sleep(0.01)
    finally:
        executor.close(identity)
    terminal = next(event for event in events if event.kind in _TERMINAL)
    assert terminal.kind is StandardEventKind.FINISHED, terminal.detail
    return dict(terminal.terminal_timing.details)


def test_the_timing_summary_reports_each_verification_step(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("XDART_PERF", "1")
    caplog.set_level(logging.INFO)
    raw, save = tmp_path / "raw", tmp_path / "append"
    raw.mkdir()
    for index in range(1, 13):
        _frame(raw, "scan", index, "txt", float(index))
    fresh = _append(tmp_path, raw, save)
    # A fresh Append on the fast writer (B-2): one science readback at close, no checkpoint or close readback.
    assert fresh["writer_finish_science_readback"] > 0, fresh
    assert "writer_finish_checkpoint_readback" not in fresh and "writer_finish_close_readback" not in fresh, fresh
    for index in range(13, 25):
        _frame(raw, "scan", index, "txt", float(index))
    epoch = _append(tmp_path, raw, save)
    # The next Append extends that output in place: its checkpoints read back what they wrote, and its close seals.
    assert epoch["writer_finish_checkpoint_readback"] > 0, epoch
    assert epoch["writer_finish_close_readback"] >= 0, epoch
    lines = [record.getMessage() for record in caplog.records
             if record.getMessage().startswith("[PERF-WRITER-FINISH]")]
    assert any("checkpoint_readback=" in line and "close_readback=" in line for line in lines), lines
