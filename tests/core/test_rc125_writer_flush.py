"""An owned epoch flushes at its checkpoint and once more after its HDF5 close (the close itself rewrites bytes after
the last flush: superblock and metadata); the terminal seal adds no third flush. Buffered: none at all."""
import hashlib
import os

import h5py
import pytest

from tests.core.test_h23_c3_composition_append import _finite_seed, _labels_intent, _r1, _readback
from xrd_tools.core.scan import Scan, ScanFrame
from xrd_tools.io import prepare_append_preflight
from xrd_tools.reduction import FrameReduction, NexusSink, NexusTerminalDisposition, ReductionPlan, ReductionResult


@pytest.mark.parametrize("durable", (True, False))
def test_in_place_epoch_fsyncs_at_its_checkpoint_and_after_its_close_under_its_admitted_policy(
        tmp_path, monkeypatch, durable):
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
    assert len(calls) == (2 if durable else 0), calls  # the checkpoint, then the closed file; no terminal flush
    assert _readback(target)[0] == (0, 1)
    assert _readback(target)[2]["state"] == "committed"


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("durable", (True, False))
def test_a_finite_overwrite_flushes_its_closed_file_once_under_its_policy(tmp_path, monkeypatch, durable):
    """The fast finite Overwrite route (no retained proofs re-read at close) is the third close path: with the flag on,
    its last flush must see the bytes the HDF5 close wrote, so the file on disk never changes after that flush (Codex
    review of bf532ec7: the route sealed its close as flushed with no flush after the close). Off: no flush at all."""
    target = tmp_path / "fast.nexus"
    sink = NexusSink(target, overwrite=True, flush_every=None, durable_fsync=durable,
                     run_configuration_provenance={"output_mode": "Overwrite", "live_mode": False})
    sink.begin(Scan("fast", []), ReductionPlan(integration_2d=None))
    assert sink._fast_regenerable, "the finite Overwrite route under test"
    flushed, real = [], os.fsync

    def fsync(descriptor):
        result = real(descriptor)
        if os.fstat(descriptor).st_ino == target.stat().st_ino:
            flushed.append(_digest(target))  # the bytes this flush made durable
        return result

    monkeypatch.setattr(os, "fsync", fsync)
    sink.write(ScanFrame(0), FrameReduction(0, result_1d=_r1(3)))
    terminal = sink.finish(ReductionResult("fast", {}, 1))
    assert terminal.disposition is NexusTerminalDisposition.COMMITTED
    assert len(flushed) == (2 if durable else 0), flushed  # the checkpoint, then the closed file
    if durable:
        assert flushed[-1] == _digest(target), "the file changed on disk after its last flush (the HDF5 close)"
    with h5py.File(target, "r") as handle:  # a fresh Overwrite carries no Append lineage: its rows say it committed
        assert tuple(int(label) for label in handle["entry/integrated_1d/frame_index"][()]) == (0,)
