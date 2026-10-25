"""A real in-place writer's close retry must flush after losing its writing handle."""
import errno
import hashlib
import os

import pytest

from tests.core.test_h23_c3_composition_append import _finite_seed, _finite_in_place_sink, _r1, _readback
from xrd_tools.core.scan import ScanFrame
from xrd_tools.io.record_writer import WriterIncomplete, WriterPhase
from xrd_tools.io.output_transaction import TransactionPhase
from xrd_tools.reduction import FrameReduction, NexusTerminalDisposition, ReductionResult


def test_an_in_place_close_retry_flushes_the_closed_file_before_committing(tmp_path, monkeypatch):
    target = tmp_path / "retry.nexus"
    _finite_seed(target, tmp_path, 1)
    sink = _finite_in_place_sink(target, tmp_path, 2)
    sink.write(ScanFrame(1), FrameReduction(1, result_1d=_r1(7)))
    writer, transaction = sink._writer, sink._transaction
    real, failed, flushed = os.fsync, [], []

    def fsync(descriptor):
        # Inject one fallible OS-boundary error, never a fake writer, seal or proof.
        # Every successful flush still runs the real fsync on the real descriptor.
        if writer._h5 is None and not failed:
            failed.append(descriptor)
            raise OSError(errno.EIO, "injected post-HDF5-close flush failure")
        result = real(descriptor)
        if writer._h5 is None:
            flushed.append(hashlib.sha256(target.read_bytes()).hexdigest())
        return result

    result = ReductionResult("in-place", {}, 2)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "fsync", fsync)
            with pytest.raises(WriterIncomplete, match="post-HDF5-close flush failure"):
                sink.finish(result)
            assert writer.phase is WriterPhase.PARTIAL and writer._h5 is None
            assert writer._close_verification_descriptor is None and writer._stream_close_attempt is not None
            assert transaction.snapshot().phase is TransactionPhase.INTEGRITY_HOLD
            terminal = sink.finish(result)
        assert terminal.disposition is NexusTerminalDisposition.COMMITTED
        assert len(failed) == 1 and len(flushed) == 1, (failed, flushed)
        assert flushed[0] == hashlib.sha256(target.read_bytes()).hexdigest()
        labels, _intensity, lineage = _readback(target)
        assert labels == (0, 1) and lineage["state"] == "committed"
    finally:
        if writer.phase not in (WriterPhase.FINISHED, WriterPhase.ABORTED):
            sink.abort(None)
