"""Real dynamic ScanSession cleanup with retryable writer-close failures."""

import pytest

from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus
from xrd_tools.io.record_writer import NexusRecordWriter
from xrd_tools.sources.image import TiffSeriesSource

from tests.xdart.scattering.test_e1b1_terminal_cleanup import _prepared_run


def _finish_close_failure(tmp_path, monkeypatch, *, failures, save_xye=True):
    from tests.xdart.scattering._e2sd_support import write_poni
    from tests.xdart.scattering.test_p1b_output_graph import _intent, _write_tiff

    raw, poni = tmp_path / "raw_0001.tif", tmp_path / "cal.poni"
    _write_tiff(raw, 1)
    write_poni(poni)
    intent = _intent(raw, tmp_path / "processed", poni)
    intent.run_options["_post_g2_output_diagnostics_v1"] = {
        "save_xye": save_xye, "durable_fsync": True,
    }
    executor, run, _admission = _prepared_run(tmp_path, intent=intent)
    close_attempts, closed_sources, remaining = [], [], [failures]
    real_close = NexusRecordWriter._close_handle

    def close_handle(writer):
        close_attempts.append(writer)
        if remaining[0]:
            if remaining[0] > 0:
                remaining[0] -= 1
            raise OSError("writer handle still open")
        return real_close(writer)

    monkeypatch.setattr(NexusRecordWriter, "_close_handle", close_handle)
    monkeypatch.setattr(
        TiffSeriesSource, "close", lambda source: closed_sources.append(source),
        raising=False,
    )
    return executor, run, close_attempts, closed_sources, remaining


@pytest.mark.parametrize("save_xye", (False, True))
def test_real_scansession_finish_failure_cannot_become_cleaned(tmp_path, monkeypatch, save_xye) -> None:
    executor, run, attempts, closed, remaining = _finish_close_failure(
        tmp_path, monkeypatch, failures=-1, save_xye=save_xye,
    )
    try:
        executor._run(run)
        terminal = executor.drain_events()[-1]
        assert terminal.kind is StandardEventKind.FAILED
        assert "writer handle still open" in terminal.detail
        assert terminal.cleanup_status is CleanupStatus.CLEANUP_PENDING
        output, session = run.output, run.session
        assert output is not None and session is not None
        assert len(closed) == 1
        assert len(attempts) >= 2 and all(writer is attempts[0] for writer in attempts)
        writer = attempts[0]
        assert writer._h5 is not None and writer._h5.id.valid
        before = len(attempts)
        assert executor.close(run.identity).cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert run.output is output and run.session is session
        assert len(attempts) == before + 1
        assert writer._h5.id.valid
    finally:
        remaining[0] = 0
        # Close first retires display custody, then retries output settlement.
        # The pending display receipt is acknowledged on the next Close.
        assert executor.close(run.identity).cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert writer._h5 is None
        assert run.output is run.session is run.sink is None
        assert executor.close(run.identity).cleanup_status is CleanupStatus.CLEANED
    assert writer._h5 is None
    assert run.output is run.session is run.sink is None
    assert len(closed) == 1
    assert writer.operation_vector().stacked_1d_rows == 1
    assert not any(event.kind is StandardEventKind.FINISHED for event in executor.drain_events())


@pytest.mark.parametrize("save_xye", (False, True))
def test_real_scansession_finish_failure_requires_writer_close_to_clean(tmp_path, monkeypatch, save_xye) -> None:
    executor, run, attempts, closed, _remaining = _finish_close_failure(
        tmp_path, monkeypatch, failures=1, save_xye=save_xye,
    )
    executor._run(run)
    terminal = executor.drain_events()[-1]
    assert terminal.kind is StandardEventKind.FAILED
    assert "writer handle still open" in terminal.detail
    assert terminal.cleanup_status is CleanupStatus.CLEANED
    assert len(attempts) == 2 and attempts[0] is attempts[1]
    assert attempts[0]._h5 is None and attempts[0].phase.value == "finished"
    assert run.output is run.session is run.sink is None
    assert len(closed) == 1
    assert executor.close(run.identity).cleanup_status is CleanupStatus.CLEANED
    assert len(attempts) == 2
    assert attempts[0].operation_vector().stacked_1d_rows == 1
    assert not any(event.kind is StandardEventKind.FINISHED for event in executor.drain_events())
