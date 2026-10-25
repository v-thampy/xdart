"""Closing an Append reuses proofs already read at its checkpoints."""
import sys

import h5py
import numpy as np

from tests.core.test_h23_c3_composition_append import (
    ItemDisposition, ResultMode, StageLedger, _LedgerFacade, _finite_seed, _labels_intent, _r1, _readback,
)
from xrd_tools.core.scan import Scan, ScanFrame
from xrd_tools.io import prepare_append_preflight
from xrd_tools.reduction import FrameReduction, NexusSink, NexusTerminalDisposition, ReductionPlan, ReductionResult


def test_five_hundred_frame_append_close_reads_no_rows(tmp_path, monkeypatch):
    target = tmp_path / "close.nexus"
    _finite_seed(target, tmp_path, 1)
    preflight = prepare_append_preflight(target, _labels_intent(tmp_path, 501, generation=1), in_place=True)
    sink = NexusSink(target, source_base=tmp_path, append_preflight=preflight, flush_every=None)
    mode = ResultMode.one_d()
    ledger = StageLedger(required_modes=(mode,), targets_by_mode={mode: (f"nexus:{target}",)})
    sink.bind_session(_LedgerFacade(ledger))  # real immutable H10 ledger at the writer boundary
    sink.begin(Scan("close", []), ReductionPlan(integration_2d=None))
    for label in range(1, 501):
        accepted = ledger.record_accepted(label)
        ledger.record_outcome(label, ItemDisposition.COMPLETED, produced=(mode,), attempt=accepted)
        ledger.record_written(label, (mode,))
        sink.write(ScanFrame(label), FrameReduction(label, result_1d=_r1(label)))
    sink.flush(force=True)
    assert sink._writer._durable_mode_proofs, "the close under test must consume retained proofs"
    reads, real = [], h5py.Dataset.__getitem__

    def getitem(dataset, key, *args, **kwargs):
        value = real(dataset, key, *args, **kwargs)
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_name == "_seal_verified_stream_close":
                reads.append(np.asarray(value).nbytes)
                break
            frame = frame.f_back
        return value

    monkeypatch.setattr(h5py.Dataset, "__getitem__", getitem)
    terminal = sink.finish(ReductionResult("close", {}, 501))
    assert terminal.disposition is NexusTerminalDisposition.COMMITTED
    assert sum(reads) == 0, sum(reads)
    assert _readback(target)[0] == tuple(range(501))
