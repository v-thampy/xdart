"""An owned epoch flushes once at its checkpoint, with no duplicate close flush."""
import os

import pytest

from tests.core.test_h23_c3_composition_append import _finite_seed, _labels_intent, _r1, _readback
from xrd_tools.core.scan import Scan, ScanFrame
from xrd_tools.io import prepare_append_preflight
from xrd_tools.reduction import FrameReduction, NexusSink, NexusTerminalDisposition, ReductionPlan, ReductionResult


@pytest.mark.parametrize("durable", (True, False))
def test_in_place_epoch_fsync_follows_its_admitted_policy_once(tmp_path, monkeypatch, durable):
    target = tmp_path / "flush.nexus"
    _finite_seed(target, tmp_path, 1)
    preflight = prepare_append_preflight(target, _labels_intent(tmp_path, 2, generation=1),
                                        in_place=True, durable_fsync=durable)
    sink = NexusSink(target, source_base=tmp_path, append_preflight=preflight,
                     durable_fsync=durable, flush_every=None)
    sink.begin(Scan("flush", []), ReductionPlan(integration_2d=None))
    calls, real = [], os.fsync

    def fsync(descriptor):
        calls.append(descriptor)
        return real(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)
    sink.write(ScanFrame(1), FrameReduction(1, result_1d=_r1(1)))
    terminal = sink.finish(ReductionResult("flush", {}, 2))
    assert terminal.disposition is NexusTerminalDisposition.COMMITTED
    assert len(calls) == int(durable), calls
    assert _readback(target)[0] == (0, 1)
    assert _readback(target)[2]["state"] == "committed"
