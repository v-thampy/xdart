"""Keep active flush recovery distinct from retryable terminal progress."""

import numpy as np
import pytest

from xrd_tools.io import read_frame_records
from xrd_tools.io.record_writer import (
    NexusRecordWriter, RecordWrite, WriterIncomplete, WriterPhase,
)
from tests.core.test_h23_record_writer import _r1


@pytest.mark.parametrize("timing_env", (None, "XDART_PERF", "XDART_PERF_QUARTILES"))
def test_finish_timings_are_opt_in_and_preserve_records(tmp_path, monkeypatch, timing_env):
    monkeypatch.delenv("XDART_PERF", raising=False)
    monkeypatch.delenv("XDART_PERF_QUARTILES", raising=False)
    if timing_env is not None:
        monkeypatch.setenv(timing_env, "1")
    writer = NexusRecordWriter(tmp_path / "timed.nexus", atomic=False, flush_every=None)
    writer.begin()
    try:
        writer.write(RecordWrite(label=1, result_1d=_r1(7)))
        assert writer.perf_snapshot() == {}
        outcome = writer.finish()
        measured = writer.perf_snapshot()
        if timing_env is None:
            assert measured == {}
        else:
            assert set(measured) == {
                "writer_finish_" + name for name in (
                    "metadata", "flush", "admission", "checkpoint", "close",
                    "replace", "resume", "receipt", "publication_drop",
                )
            }
            assert all(value >= 0 for value in measured.values())
            assert measured["writer_finish_close"] > 0
        measured["caller_edit"] = 123
        assert "caller_edit" not in writer.perf_snapshot()
        before = writer.perf_snapshot()
        assert writer.finish() == outcome
        assert writer.perf_snapshot() == before
        assert writer._h5 is None
        (record,) = read_frame_records(writer.target)
        assert record.label == 1
        np.testing.assert_array_equal(record.results_1d["default"].intensity_1d, np.full(4, 7))
    finally:
        if writer._h5 is not None:
            writer.abort()


@pytest.mark.parametrize("recovery", ("flush", "finish"))
def test_nonterminal_flush_failure_keeps_its_recovery_owner(tmp_path, monkeypatch, recovery):
    writer = NexusRecordWriter(tmp_path / "flush-retry.nexus", atomic=False, flush_every=None)
    assert not writer.finalization_started
    writer.begin()
    writer.write(RecordWrite(label=1, result_1d=_r1(7), metadata={"temperature": 12.5}))
    real_flush = writer._flush_handle
    attempts = []

    def flush_once():
        attempts.append(writer)
        if len(attempts) == 1:
            raise OSError("nonterminal flush failed")
        return real_flush()

    monkeypatch.setattr(writer, "_flush_handle", flush_once)
    try:
        with pytest.raises(WriterIncomplete) as caught:
            writer.flush(force=True)
        assert caught.value.outcome.pending_owner == "flush"
        assert writer.phase is WriterPhase.PARTIAL
        assert not writer.finalization_started
        if recovery == "flush":
            writer.flush(force=True)
            assert writer.phase is WriterPhase.ACTIVE
            assert not writer.finalization_started
            # Active flush recovery permits another real row before finish.
            writer.write(RecordWrite(label=2, result_1d=_r1(9)))
        writer.finish()
        assert writer.finalization_started
        assert writer.phase is WriterPhase.FINISHED
        assert writer._h5 is None
        records = read_frame_records(writer.target)
        assert tuple(record.label for record in records) == ((1, 2) if recovery == "flush" else (1,))
        for record, expected in zip(records, (7, 9)):
            np.testing.assert_array_equal(record.results_1d["default"].intensity_1d, np.full(4, expected))
        assert records[0].results_1d["default"].metadata_numeric == {"temperature": 12.5}
        assert writer.operation_vector().stacked_1d_rows == len(records)
    finally:
        if writer._h5 is not None:
            writer.abort()
