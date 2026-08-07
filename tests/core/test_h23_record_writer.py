"""Finite H23-C2 oracle for the shared headless NeXus writer lifecycle."""

from __future__ import annotations

import hashlib
import importlib
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.core.metadata import ScanMetadata
from xrd_tools.io import read_frame_records
from xrd_tools.io.nexus import (
    open_nexus_writer,
    read_scan,
    write_integrated_stack,
    write_scan_metadata,
)
from xrd_tools.io.nexus_record import make_thumbnail_array, stamp_source_base
from xrd_tools.io.schema import PRIMARY_MODE_ATTR
from xrd_tools.session.stage_accounting import ResultMode, StageReceipt


PARENT_C1_SHA256 = "eac1d17129339108be0926a1ce3869dc613f2d6d7c749e46244d552e879e35ff"
PARENT_C1_TEST_SHA256 = "127923d0d0730e0922d741f4d78d57e2d996d2fb9847ec56d4133ad4bf24b9d3"


def _api():
    return importlib.import_module("xrd_tools.io.record_writer")


def _r1(value: float, *, n: int = 4) -> IntegrationResult1D:
    return IntegrationResult1D(
        radial=np.linspace(0.1, 1.0, n),
        intensity=np.full(n, value, dtype=float),
        sigma=np.full(n, value / 10, dtype=float),
        unit="q_A^-1",
    )


def _r2(value: float, *, nq: int = 4, nchi: int = 3) -> IntegrationResult2D:
    raw = np.arange(nq * nchi, dtype=float).reshape(nq, nchi) + value
    return IntegrationResult2D(
        radial=np.linspace(0.1, 1.0, nq),
        azimuthal=np.linspace(-1.0, 1.0, nchi),
        intensity=raw,
        sigma=raw / 10,
        unit="q_A^-1",
        azimuthal_unit="chi_deg",
    )


class _Facade:
    def __init__(self, target: Path) -> None:
        self.target = f"nexus:{target}"
        self.revisions: dict[tuple[int, ResultMode], int] = {}
        self.durable: list[StageReceipt] = []
        self.dropped: list[tuple[int, ResultMode, int]] = []

    def set_revision(self, label: int, mode: ResultMode, revision: int) -> None:
        self.revisions[(int(label), mode)] = int(revision)

    def targets_for(self, mode: ResultMode) -> frozenset[str]:
        return frozenset((self.target, "xye:/not-owned-by-nexus.xye"))

    def capture_receipt(self, label: int, mode: ResultMode, target: str) -> StageReceipt:
        return StageReceipt(int(label), mode, self.revisions[(int(label), mode)], target)

    def commit_durable(self, receipts) -> None:
        batch = tuple(receipts)
        for receipt in batch:
            assert receipt.revision == self.revisions[(receipt.label, receipt.mode)]
            assert receipt.target == self.target
        self.durable.extend(batch)

    def commit_publication_drop(self, label, mode, expected_revision) -> None:
        current = self.revisions[(int(label), mode)]
        if int(expected_revision) > current:
            raise ValueError("future publication drop")
        if int(expected_revision) == current:
            self.dropped.append((int(label), mode, int(expected_revision)))


class _TraceLock(AbstractContextManager):
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace

    def __enter__(self):
        self.trace.append("lock-enter")
        return self

    def __exit__(self, *exc):
        self.trace.append("lock-exit")
        return False


class _TracePool:
    def __init__(self, trace: list[str], *, fail_resume: bool = False) -> None:
        self.trace = trace
        self.fail_resume = fail_resume

    def pause(self, path) -> None:
        self.trace.append(f"pause:{Path(path).name}")

    def resume(self, path) -> None:
        self.trace.append(f"resume:{Path(path).name}")
        if self.fail_resume:
            raise OSError("resume failed")


def test_shared_lifecycle_owns_nexussink_handle_cursor_and_terminal_state():
    from xrd_tools.reduction import core as reduction_core
    from xrd_tools.reduction.core import NexusSink

    sink_fields = set(NexusSink.__dataclass_fields__)
    assert {"_h5", "_n_written", "_active_path", "_tmp_path"}.isdisjoint(sink_fields)
    assert "_writer" in sink_fields
    rw = _api()
    assert rw.NexusRecordWriter.__module__ == "xrd_tools.io.record_writer"
    assert {"begin", "write", "flush", "finish", "abort"} <= set(
        rw.NexusRecordWriter.__dict__
    )
    assert "_ORIGINAL_WRITE_NEXUS_FRAME" not in vars(reduction_core)


def test_h10_order_and_receipt_commit_follow_positive_flush(monkeypatch, tmp_path):
    rw = _api()
    target = tmp_path / "ordered.nexus"
    trace: list[str] = []
    facade = _Facade(target)
    mode = ResultMode.one_d("default")
    facade.set_revision(7, mode, 1)
    real_write = rw.write_integrated_stack

    def traced_write(*args, **kwargs):
        trace.append("write")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(rw, "write_integrated_stack", traced_write)
    writer = rw.NexusRecordWriter(
        target, atomic=False, flush_every=None,
        file_lock=_TraceLock(trace), pool=_TracePool(trace),
    )
    writer.bind_session(facade)
    writer.begin(primary_mode_1d="default", primary_mode_2d="default")
    writer.write(rw.RecordWrite(label=7, result_1d=_r1(7)))
    assert facade.durable == []
    writer.flush(force=True)
    assert [(r.label, r.mode, r.revision) for r in facade.durable] == [(7, mode, 1)]
    writer.finish()

    assert trace[:3] == ["lock-enter", "pause:ordered.nexus", "lock-exit"]
    write_at = trace.index("write")
    assert trace[write_at - 1] == "lock-enter" and trace[write_at + 1] == "lock-exit"
    resume_at = trace.index("resume:ordered.nexus")
    assert trace[resume_at - 1] == "lock-enter" and trace[resume_at + 1] == "lock-exit"


def test_nonterminal_flush_checkpoints_exact_metadata_once(tmp_path):
    rw = _api()
    target = tmp_path / "checkpoint-metadata.nexus"
    writer = rw.NexusRecordWriter(target, atomic=False, flush_every=None)
    writer.begin()
    writer.write(rw.RecordWrite(
        label=7,
        result_1d=_r1(7),
        metadata={"temperature": 12.5, "status": "collecting"},
    ))

    writer.flush(force=True)

    (record,) = read_frame_records(target)
    view = record.results_1d["default"]
    assert dict(view.metadata_raw) == {
        "status": "collecting",
        "temperature": np.float32(12.5),
    }
    assert dict(view.metadata_numeric) == {"temperature": 12.5}
    assert writer.operation_vector().indexed_metadata_rows == 1

    outcome = writer.finish()
    assert outcome.operation_vector.indexed_metadata_rows == 1


def test_checkpoint_metadata_upsert_failure_retains_exact_retry(monkeypatch, tmp_path):
    rw = _api()
    target = tmp_path / "checkpoint-metadata-retry.nexus"
    facade = _Facade(target)
    mode = ResultMode.one_d("default")
    facade.set_revision(7, mode, 1)
    writer = rw.NexusRecordWriter(target, atomic=False, flush_every=None)
    writer.bind_session(facade)
    writer.begin()
    writer.write(rw.RecordWrite(
        label=7,
        result_1d=_r1(7),
        metadata={"temperature": 12.5},
    ))
    real_upsert = rw.upsert_scan_metadata
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("metadata upsert failed")
        return real_upsert(*args, **kwargs)

    monkeypatch.setattr(rw, "upsert_scan_metadata", fail_once)
    with pytest.raises(rw.WriterIncomplete) as caught:
        writer.flush(force=True)
    assert caught.value.outcome.pending_owner == "flush"
    assert writer._pending_metadata_labels == {7}
    assert facade.durable == []

    writer.flush(force=True)
    assert [(receipt.label, receipt.mode, receipt.revision)
            for receipt in facade.durable] == [(7, mode, 1)]
    assert read_frame_records(target)[0].results_1d["default"].metadata_numeric == {
        "temperature": 12.5,
    }
    writer.finish()


@pytest.mark.parametrize("boundary", ["flush", "close", "resume"])
def test_flush_close_and_resume_failures_withhold_receipts(monkeypatch, tmp_path, boundary):
    rw = _api()
    target = tmp_path / f"{boundary}.nexus"
    trace: list[str] = []
    facade = _Facade(target)
    mode = ResultMode.one_d("default")
    facade.set_revision(1, mode, 1)
    pool = _TracePool(trace, fail_resume=boundary == "resume")
    writer = rw.NexusRecordWriter(
        target, atomic=False, flush_every=None,
        file_lock=_TraceLock(trace), pool=pool,
    )
    writer.bind_session(facade)
    writer.begin()
    writer.write(rw.RecordWrite(label=1, result_1d=_r1(1)))
    if boundary == "flush":
        monkeypatch.setattr(writer, "_flush_handle", lambda: (_ for _ in ()).throw(OSError("flush failed")))
    elif boundary == "close":
        monkeypatch.setattr(writer, "_close_handle", lambda: (_ for _ in ()).throw(OSError("close failed")))
    with pytest.raises(rw.WriterIncomplete) as caught:
        writer.finish()
    assert facade.durable == []
    assert caught.value.outcome.phase is rw.WriterPhase.PARTIAL
    assert caught.value.outcome.pending_owner == boundary


def test_automatic_cadence_flush_keeps_exact_flush_owner(monkeypatch, tmp_path):
    rw = _api()
    target = tmp_path / "cadence-owner.nexus"
    writer = rw.NexusRecordWriter(target, atomic=False, flush_every=1)
    writer.begin()
    monkeypatch.setattr(
        writer,
        "_flush_handle",
        lambda: (_ for _ in ()).throw(OSError("flush failed")),
    )
    with pytest.raises(rw.WriterIncomplete) as caught:
        writer.write(rw.RecordWrite(label=1, result_1d=_r1(1)))
    assert caught.value.outcome.pending_owner == "flush"
    writer.abort()


def test_stale_pending_is_superseded_and_drop_is_explicit(tmp_path):
    rw = _api()
    target = tmp_path / "receipts.nexus"
    facade = _Facade(target)
    mode_1d = ResultMode.one_d("default")
    mode_2d = ResultMode.two_d("default")
    facade.set_revision(4, mode_1d, 1)
    facade.set_revision(4, mode_2d, 1)
    writer = rw.NexusRecordWriter(target, atomic=False, flush_every=None)
    writer.bind_session(facade)
    writer.begin()
    writer.write(rw.RecordWrite(label=4, result_1d=_r1(1), result_2d=_r2(1)))
    facade.set_revision(4, mode_1d, 2)
    writer.flush(force=True)
    assert {(r.mode, r.revision) for r in facade.durable} == {(mode_2d, 1)}
    writer.write(rw.RecordWrite(label=4, result_1d=_r1(2)))
    writer.mark_publication_dropped(4, mode_1d, expected_revision=1)
    assert facade.dropped == []
    writer.flush(force=True)
    assert (mode_1d, 2) in {(r.mode, r.revision) for r in facade.durable}
    facade.set_revision(4, mode_1d, 3)
    writer.mark_publication_dropped(4, mode_1d, expected_revision=3)
    assert facade.dropped == []
    writer.flush(force=True)
    assert facade.dropped == [(4, mode_1d, 3)]
    writer.finish()


def test_current_publication_drop_removes_only_exact_mode_row_and_keeps_gi_child(
    tmp_path,
):
    rw = _api()
    target = tmp_path / "drop-current-mode.nexus"
    facade = _Facade(target)
    primary = ResultMode.one_d("q_ip")
    sibling = ResultMode.one_d("q_oop")
    facade.set_revision(0, primary, 1)
    facade.set_revision(0, sibling, 1)
    writer = rw.NexusRecordWriter(
        target, atomic=False, flush_every=None, complete_record=False,
    )
    writer.bind_session(facade)
    writer.begin(primary_mode_1d="q_ip")
    writer.write(rw.RecordWrite(label=0, result_1d=_r1(10), mode_1d="q_ip"))
    writer.write(rw.RecordWrite(label=0, result_1d=_r1(20), mode_1d="q_oop"))

    facade.set_revision(0, primary, 2)
    writer.mark_publication_dropped(0, primary, expected_revision=2)
    writer.flush(force=True)
    assert facade.dropped == [(0, primary, 2)]
    with h5py.File(target, "r") as handle:
        top = handle["entry/integrated_1d"]
        assert tuple(top["frame_index"][()]) == ()
        child = top["q_oop"]
        assert tuple(child["frame_index"][()]) == (0,)
        np.testing.assert_allclose(child["intensity"][0], 20)
    writer.finish()


def test_publication_drop_row_removal_failure_is_partial_without_h10_drop(
    tmp_path,
):
    rw = _api()
    target = tmp_path / "drop-removal-failure.nexus"
    facade = _Facade(target)
    mode = ResultMode.one_d()
    facade.set_revision(0, mode, 1)
    writer = rw.NexusRecordWriter(target, atomic=False, flush_every=None)
    writer.bind_session(facade)
    writer.begin()
    writer.write(rw.RecordWrite(label=0, result_1d=_r1(1)))
    facade.set_revision(0, mode, 2)
    writer._drop_mode_row = lambda *_a, **_k: (_ for _ in ()).throw(
        OSError("row removal failed")
    )
    with pytest.raises(rw.WriterIncomplete) as excinfo:
        writer.mark_publication_dropped(0, mode, expected_revision=2)
    assert excinfo.value.outcome.pending_owner == "publication-drop"
    assert facade.dropped == []
    writer.abort()


def test_publication_drop_waits_for_positive_absence_checkpoint(
    tmp_path,
):
    rw = _api()
    target = tmp_path / "drop-absence-checkpoint.nexus"
    facade = _Facade(target)
    mode = ResultMode.one_d()
    facade.set_revision(0, mode, 1)
    writer = rw.NexusRecordWriter(target, atomic=False, flush_every=None)
    writer.bind_session(facade)
    writer.begin()
    writer.write(rw.RecordWrite(label=0, result_1d=_r1(1)))
    facade.set_revision(0, mode, 2)
    writer.mark_publication_dropped(0, mode, expected_revision=2)
    assert facade.dropped == []
    writer._verify_absent_mode_row = (
        lambda *_a, **_k: (_ for _ in ()).throw(
            rw.WriterStateError("absence observation failed")
        )
    )
    with pytest.raises(rw.WriterIncomplete) as excinfo:
        writer.flush(force=True)
    assert excinfo.value.outcome.pending_owner == "flush"
    assert facade.dropped == []
    writer.abort()


def test_publication_drop_compacts_middle_row_and_preserves_cursor_reuse(tmp_path):
    rw = _api()
    target = tmp_path / "drop-middle-row.nexus"
    facade = _Facade(target)
    mode = ResultMode.one_d()
    writer = rw.NexusRecordWriter(
        target, atomic=False, flush_every=None, complete_record=False,
    )
    writer.bind_session(facade)
    writer.begin()
    for label in (0, 1, 2):
        facade.set_revision(label, mode, 1)
        writer.write(rw.RecordWrite(label=label, result_1d=_r1(label + 1)))
    writer.flush(force=True)

    facade.set_revision(1, mode, 2)
    writer.mark_publication_dropped(1, mode, expected_revision=2)
    writer.flush(force=True)
    with h5py.File(target, "r") as handle:
        group = handle["entry/integrated_1d"]
        assert tuple(group["frame_index"][()]) == (0, 2)
        np.testing.assert_allclose(group["intensity"][0], 1)
        np.testing.assert_allclose(group["intensity"][1], 3)

    facade.set_revision(2, mode, 2)
    writer.write(rw.RecordWrite(label=2, result_1d=_r1(22)))
    writer.flush(force=True)
    with h5py.File(target, "r") as handle:
        group = handle["entry/integrated_1d"]
        assert tuple(group["frame_index"][()]) == (0, 2)
        np.testing.assert_allclose(group["intensity"][1], 22)
    writer.finish()


def test_cross_mode_and_source_prevalidation_precedes_any_authoritative_mutation(tmp_path):
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.reduction.core import NexusSink, ReductionPlan, ReductionResult

    target = tmp_path / "atomic-validation.nexus"
    frame = ScanFrame(0, image=np.ones((2, 2)), source_path=tmp_path / "raw-0.h5")
    scan = Scan("atomic", [frame])
    sink = NexusSink(target, atomic=False, overwrite=True, flush_every=1)
    sink.begin(scan, ReductionPlan())
    sink.write(frame, type("Reduction", (), {
        "frame_index": 0, "result_1d": _r1(0), "result_2d": _r2(0),
        "mode_1d": "default", "mode_2d": "default", "metadata": {},
    })())
    with h5py.File(target, "r") as h5:
        before = np.asarray(h5["entry/integrated_1d/intensity"][0]).copy()
    bad_2d = _r2(9, nq=5, nchi=3)
    with pytest.raises(ValueError):
        sink.write(frame, type("Reduction", (), {
            "frame_index": 0, "result_1d": _r1(9), "result_2d": bad_2d,
            "mode_1d": "default", "mode_2d": "default", "metadata": {},
        })())
    sink.abort(ReductionResult("atomic", {}, 1, failed=True))
    assert not target.exists()
    partial = Path(sink._transaction.snapshot().partial_path)
    with h5py.File(partial, "r") as h5:
        np.testing.assert_array_equal(h5["entry/integrated_1d/intensity"][0], before)


def test_dirty_replacement_updates_exact_source_record_and_transposes_2d_once(tmp_path):
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.reduction.core import NexusSink, ReductionPlan, ReductionResult

    target = tmp_path / "replace.nexus"
    old_frame = ScanFrame(
        2, image=np.array([[0.0, 1.0], [2.0, 3.0]]),
        source_path=tmp_path / "old.h5", source_frame_index=1,
        metadata={"timestamp": "old"},
    )
    scan = Scan("replace", [old_frame])
    sink = NexusSink(target, atomic=False, overwrite=True, flush_every=None)
    sink.begin(scan, ReductionPlan())
    sink.write(old_frame, type("Reduction", (), {
        "frame_index": 2, "result_1d": None, "result_2d": _r2(1),
        "mode_1d": "default", "mode_2d": "default", "metadata": {},
    })())
    replacement = _r2(20)
    new_frame = ScanFrame(
        2, image=np.array([[4.0, 5.0], [6.0, 7.0]]),
        source_path=tmp_path / "new.h5", source_frame_index=8,
        metadata={"timestamp": "new"},
    )
    sink.replace(new_frame, type("Reduction", (), {
        "frame_index": 2, "result_1d": None, "result_2d": replacement,
        "mode_1d": "default", "mode_2d": "default", "metadata": {},
    })())
    sink.finish(ReductionResult("replace", {}, 1))
    with h5py.File(target, "r") as h5:
        np.testing.assert_allclose(h5["entry/integrated_2d/intensity"][0], replacement.intensity.T)
        frame = h5["entry/frames/frame_0002"]
        assert frame["timestamp"][()].decode() == "new"
        assert frame["source/frame_index"][()] == 8
        assert frame["source/path"].asstr()[()] == str(tmp_path / "new.h5")


@pytest.mark.parametrize(
    "mutation",
    ("source_absent", "source_path", "source_index", "snapshot",
     "mask_flag", "mask_content"),
)
def test_ordinary_existing_row_write_requires_complete_provenance_identity(
    tmp_path, mutation,
):
    rw = _api()
    target = tmp_path / f"identity-{mutation}.nexus"
    source = tmp_path / "source.h5"
    source.write_bytes(b"source")
    snapshot = {
        "adapter_id": "nexus_hdf5", "size": source.stat().st_size,
        "mtime_ns": source.stat().st_mtime_ns, "frame_count": 1,
        "dataset_path": "/entry/data/data", "self_contained": True,
    }
    base = dict(
        label=0, result_1d=_r1(1), source_path=source,
        source_frame_index=3, source_snapshot=snapshot,
        thumbnail=np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32),
        thumbnail_mask=np.array([[False, True], [False, False]], dtype=bool),
        thumbnail_mask_baked=False, mask_baked=False,
    )
    writer = rw.NexusRecordWriter(target, atomic=False, flush_every=None)
    writer.begin()
    writer.write(rw.RecordWrite(**base))
    writer.flush(force=True)
    before = target.read_bytes()
    changed = dict(base)
    if mutation == "source_absent":
        changed.update(
            source_path=None, source_frame_index=0, source_snapshot={},
        )
    elif mutation == "source_path":
        changed["source_path"] = tmp_path / "other.h5"
    elif mutation == "source_index":
        changed["source_frame_index"] = 4
    elif mutation == "snapshot":
        changed["source_snapshot"] = {**snapshot, "frame_count": 2}
    elif mutation == "mask_flag":
        changed["mask_baked"] = True
    else:
        changed["thumbnail_mask"] = np.array(
            [[True, True], [False, False]], dtype=bool,
        )
    with pytest.raises(rw.WriterStateError):
        writer.write(rw.RecordWrite(**changed))
    assert target.read_bytes() == before
    writer.abort()


@pytest.mark.parametrize(
    "kwargs",
    (
        {"source_path": "raw.h5", "source_snapshot": {"size": -1}},
        {"source_path": "raw.h5", "source_snapshot": {"unknown": "x"}},
        {"source_path": "raw.h5", "source_snapshot": {"self_contained": 1}},
        {"thumbnail_mask": np.ones((2, 2), dtype=np.uint8)},
    ),
)
def test_record_provenance_rejects_malformed_values_before_writer_mutation(
    tmp_path, kwargs,
):
    rw = _api()
    target = tmp_path / "malformed.nexus"
    writer = rw.NexusRecordWriter(target, atomic=False, flush_every=None)
    writer.begin()
    writer._h5.flush()
    before = target.read_bytes()
    with pytest.raises((TypeError, ValueError)):
        writer.write(rw.RecordWrite(label=0, result_1d=_r1(1), **kwargs))
    writer._h5.flush()
    assert target.read_bytes() == before
    writer.abort()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("replace_existing", "yes"),
        ("write_frame_record", 1),
        ("mask_baked", "false"),
        ("thumbnail_mask_baked", 0),
    ),
)
def test_record_write_boolean_schema_refuses_truthiness(field, value):
    rw = _api()
    with pytest.raises((TypeError, ValueError)):
        rw.RecordWrite(label=0, result_1d=_r1(1), **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("label", 0.0),
        ("label", True),
        ("source_frame_index", 1.0),
        ("source_frame_index", 1.5),
        ("source_frame_index", True),
        ("source_snapshot", {"size": 1.0}),
        ("source_snapshot", {"mtime_ns": True}),
        ("source_snapshot", {"frame_count": "1"}),
        ("source_snapshot", {"self_contained": "false"}),
    ),
)
def test_record_write_integral_and_snapshot_schema_refuses_coercion(
    tmp_path, field, value,
):
    rw = _api()
    kwargs = {"label": 0, "result_1d": _r1(1), "source_path": tmp_path / "raw.h5"}
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError)):
        rw.RecordWrite(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"source_frame_index": 1},
        {"source_frame_index": -1},
        {"source_frame_index": 0.5},
        {"source_frame_index": True},
        {"source_snapshot": {"size": 1}},
    ),
)
def test_absent_source_accepts_only_the_canonical_default_selector(kwargs):
    rw = _api()
    with pytest.raises((TypeError, ValueError)):
        rw.RecordWrite(label=0, result_1d=_r1(1), **kwargs)


def test_mode_only_write_cannot_bypass_authoritative_source_identity(tmp_path):
    rw = _api()
    target = tmp_path / "mode-only-source.nexus"
    source = tmp_path / "source.h5"
    source.write_bytes(b"source")
    writer = rw.NexusRecordWriter(target, atomic=False, flush_every=None)
    writer.begin()
    writer.write(rw.RecordWrite(
        label=0, result_1d=_r1(1), source_path=source, source_frame_index=2,
    ))
    writer.flush(force=True)
    before = target.read_bytes()
    with pytest.raises(rw.WriterStateError):
        writer.write(rw.RecordWrite(
            label=0, result_1d=_r1(2), write_frame_record=False,
            source_path=tmp_path / "foreign.h5", source_frame_index=9,
        ))
    assert target.read_bytes() == before
    writer.abort()


def test_explicit_replace_requires_an_existing_frame_label(tmp_path):
    rw = _api()
    target = tmp_path / "replace-absent.nexus"
    writer = rw.NexusRecordWriter(target, atomic=False, flush_every=None)
    writer.begin()
    writer._h5.flush()
    before = target.read_bytes()
    with pytest.raises(rw.WriterStateError):
        writer.write(rw.RecordWrite(
            label=4, result_1d=_r1(1), replace_existing=True,
        ))
    writer._h5.flush()
    assert target.read_bytes() == before
    writer.abort()


def test_mode_only_write_requires_an_existing_frame_label(tmp_path):
    rw = _api()
    target = tmp_path / "mode-only-absent.nexus"
    writer = rw.NexusRecordWriter(target, atomic=False, flush_every=None)
    writer.begin()
    writer._h5.flush()
    before = target.read_bytes()
    with pytest.raises(rw.WriterStateError):
        writer.write(rw.RecordWrite(
            label=4, result_1d=_r1(1), write_frame_record=False,
        ))
    writer._h5.flush()
    assert target.read_bytes() == before
    writer.abort()


def test_frame_provenance_corruption_is_read_back_before_durability(tmp_path):
    rw = _api()
    target = tmp_path / "pre-durable-corruption.nexus"
    writer = rw.NexusRecordWriter(target, atomic=False, flush_every=None)
    writer.begin()
    writer.write(rw.RecordWrite(
        label=0, result_1d=_r1(1),
        thumbnail=np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32),
        thumbnail_mask=np.array([[False, True], [False, False]], dtype=bool),
    ))
    mask = writer._h5["entry/frames/frame_0000/thumbnail_mask"]
    mask[0, 0] = True
    with pytest.raises(rw.WriterIncomplete, match="durability readback"):
        writer.flush(force=True)
    writer.abort()


def test_finish_replace_failure_preserves_typed_partial_artifact(monkeypatch, tmp_path):
    rw = _api()
    target = tmp_path / "partial.nexus"
    writer = rw.NexusRecordWriter(target, atomic=True, flush_every=None, replace_attempts=2)
    writer.begin()
    writer.write(rw.RecordWrite(label=0, result_1d=_r1(0)))
    monkeypatch.setattr(rw.os, "replace", lambda *args: (_ for _ in ()).throw(PermissionError("busy")))
    with pytest.raises(rw.WriterIncomplete) as caught:
        writer.finish()
    outcome = caught.value.outcome
    assert outcome.phase is rw.WriterPhase.PARTIAL
    assert outcome.pending_owner == "replace"
    assert outcome.partial_path is not None and outcome.partial_path.exists()
    assert not target.exists()


def _seed_target(path: Path, n: int) -> None:
    results_1d = [_r1(float(i), n=3) for i in range(n)]
    results_2d = [_r2(float(i), nq=3, nchi=2) for i in range(n)]
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        write_integrated_stack(
            entry, frame_indices=list(range(n)),
            results_1d=results_1d, results_2d=results_2d,
        )
        scan_data = pd.DataFrame(
            {"temperature": np.arange(n, dtype=float), "tag": [f"f{i}" for i in range(n)]},
            index=range(n),
        )
        write_scan_metadata(entry, scan_data, list(range(n)))


def test_atomic_sparse_update_preserves_every_untouched_row(tmp_path):
    rw = _api()
    target = tmp_path / "atomic-sparse.nexus"
    _seed_target(target, 4)
    with h5py.File(target, "r") as h5:
        before = np.asarray(h5["entry/integrated_1d/intensity"][()]).copy()
    writer = rw.NexusRecordWriter(
        target, atomic=True, overwrite=False, flush_every=None,
        complete_record=False,
    )
    writer.begin()
    writer.write(rw.RecordWrite(label=1, result_1d=_r1(99, n=3)))
    writer.finish()
    with h5py.File(target, "r") as h5:
        assert Path(h5.attrs["file_name"]).resolve() == target.resolve()
        labels = np.asarray(h5["entry/integrated_1d/frame_index"][()])
        rows = np.asarray(h5["entry/integrated_1d/intensity"][()])
    np.testing.assert_array_equal(labels, np.arange(4))
    np.testing.assert_array_equal(rows[[0, 2, 3]], before[[0, 2, 3]])
    np.testing.assert_array_equal(rows[1], _r1(99, n=3).intensity)


def test_existing_primary_mode_cannot_be_reinterpreted(tmp_path):
    rw = _api()
    target = tmp_path / "mode-mismatch.nexus"
    h5 = open_nexus_writer(target, overwrite=True)
    try:
        write_integrated_stack(
            h5["entry"], frame_indices=list(range(4)),
            results_1d=[_r1(float(i)) for i in range(4)],
            primary_mode_1d="q_ip",
        )
    finally:
        h5.close()
    with h5py.File(target, "r") as h5:
        before = h5["entry/integrated_1d"].attrs[PRIMARY_MODE_ATTR]
    writer = rw.NexusRecordWriter(target, atomic=False, flush_every=None)
    with pytest.raises(rw.WriterIncomplete) as caught:
        writer.begin(primary_mode_1d="q_oop")
    assert caught.value.outcome.pending_owner == "begin"
    writer.abort()
    with h5py.File(target, "r") as h5:
        assert h5["entry/integrated_1d"].attrs[PRIMARY_MODE_ATTR] == before


def test_source_base_mismatch_refuses_before_header_mutation(tmp_path):
    rw = _api()
    target = tmp_path / "source-base-mismatch.nexus"
    old = ScanMetadata("old", 12.0, 1.0, {}, {})
    new = ScanMetadata("new", 13.0, 0.9, {}, {})
    h5 = open_nexus_writer(target, metadata=old, overwrite=True)
    try:
        stamp_source_base(h5["entry"], tmp_path / "project-a")
    finally:
        h5.close()
    writer = rw.NexusRecordWriter(
        target, atomic=False, flush_every=None,
        source_base=tmp_path / "project-b",
    )
    with pytest.raises(rw.WriterIncomplete) as caught:
        writer.begin(metadata=new)
    assert caught.value.outcome.pending_owner == "begin"
    writer.abort()
    with h5py.File(target, "r") as h5:
        assert h5["entry"].attrs["scan_id"] == "old"


def test_malformed_indexed_group_refuses_before_header_or_row_mutation(tmp_path):
    rw = _api()
    target = tmp_path / "missing-frame-index.nexus"
    _seed_target(target, 4)
    with h5py.File(target, "r+") as h5:
        h5["entry"].attrs["scan_id"] = "original"
        del h5["entry/scan_data/frame_index"]
        before = np.asarray(h5["entry/integrated_1d/intensity"][1]).copy()
        old_scan_id = h5["entry"].attrs["scan_id"]
    writer = rw.NexusRecordWriter(target, atomic=False, flush_every=None)
    with pytest.raises(rw.WriterIncomplete) as caught:
        writer.begin(metadata=ScanMetadata("replacement", 12.0, 1.0, {}, {}))
    assert caught.value.outcome.pending_owner == "begin"
    writer.abort()
    with h5py.File(target, "r") as h5:
        assert h5["entry"].attrs["scan_id"] == old_scan_id
        np.testing.assert_array_equal(
            h5["entry/integrated_1d/intensity"][1], before,
        )


def test_existing_empty_hdf5_file_gets_complete_empty_cursor_map(tmp_path):
    rw = _api()
    target = tmp_path / "empty-existing.nexus"
    with h5py.File(target, "w"):
        pass
    writer = rw.NexusRecordWriter(
        target, atomic=False, overwrite=False, flush_every=None,
        complete_record=False,
    )
    writer.begin()
    writer.write(rw.RecordWrite(label=3, result_1d=_r1(3)))
    writer.finish()
    with h5py.File(target, "r") as h5:
        np.testing.assert_array_equal(
            h5["entry/integrated_1d/frame_index"][()], [3],
        )


def test_stale_metadata_cursor_refuses_before_integrated_mutation(tmp_path):
    rw = _api()
    target = tmp_path / "metadata-cursor.nexus"
    _seed_target(target, 4)
    writer = rw.NexusRecordWriter(target, atomic=False, flush_every=None)
    writer.begin()
    writer._row_cursors["scan_data"][1] = 2
    with h5py.File(target, "r") as h5:
        before = np.asarray(h5["entry/integrated_1d/intensity"][1]).copy()
    with pytest.raises(rw.WriterStateError, match="cursor"):
        writer.write(rw.RecordWrite(
            label=1, result_1d=_r1(99), metadata={"temperature": 99.0},
        ))
    writer.abort()
    with h5py.File(target, "r") as h5:
        np.testing.assert_array_equal(
            h5["entry/integrated_1d/intensity"][1], before,
        )


def _dirty_run(path: Path, n: int):
    rw = _api()
    _seed_target(path, n)
    labels = (1, n // 2, n - 2)
    writer = rw.NexusRecordWriter(path, atomic=False, flush_every=None)
    writer.begin()
    writer.reset_operation_vector()
    for label in labels:
        writer.write(rw.RecordWrite(
            label=label, result_1d=_r1(10000 + label, n=3),
            result_2d=_r2(10000 + label, nq=3, nchi=2),
            source_path=path.parent / f"raw-{label}.h5",
            timestamp=f"dirty-{label}",
            thumbnail=make_thumbnail_array(np.full((2, 2), label, dtype=float)),
            metadata={"temperature": 10000.0 + label, "tag": f"dirty-{label}"},
        ))
    outcome = writer.finish(rw.WriterFinalization(
        provenance_config={"dirty_labels": list(labels)},
        provenance_inputs=None,
        date="2026-08-05T00:00:00Z",
        host="",
    ))
    return labels, outcome.operation_vector


def test_n64_n4096_k3_full_update_operation_vectors_are_identical(tmp_path):
    labels_64, small = _dirty_run(tmp_path / "n64.nexus", 64)
    labels_4096, large = _dirty_run(tmp_path / "n4096.nexus", 4096)
    assert len(labels_64) == len(labels_4096) == 3
    assert asdict(small) == asdict(large)
    assert small.prepare_rows == 3
    assert small.stacked_1d_rows == small.stacked_2d_rows == 3
    assert small.indexed_metadata_rows == small.source_record_rows == 3
    assert small.provenance_boundaries == 1
    assert small.frame_index_scan_rows == 0
    assert small.untouched_hydration_rows == small.harvested_source_rows == 0
    assert max(
        small.stacked_1d_rows, small.stacked_2d_rows,
        small.indexed_metadata_rows, small.source_record_rows,
    ) <= 4


def test_known_row_cursor_refuses_stale_position_before_mutation(tmp_path):
    rw = _api()
    target = tmp_path / "stale-cursor.nexus"
    _seed_target(target, 8)
    writer = rw.NexusRecordWriter(target, atomic=False, flush_every=None)
    writer.begin()
    writer._row_cursors["integrated_1d"][3] = 4
    before = target.read_bytes()
    with pytest.raises(rw.WriterStateError, match="cursor"):
        writer.write(rw.RecordWrite(label=3, result_1d=_r1(99, n=3)))
    writer.flush(force=True)
    assert target.read_bytes() == before
    writer.abort()


def test_c1_is_mounted_through_qt_free_writer_binding():
    root = Path(__file__).resolve().parents[2]
    writer_source = (root / "src/xrd_tools/io/record_writer.py").read_text()
    assert "WriterTransactionBinding" in writer_source
    assert "seal_stream_checkpoint" in writer_source
    assert all(name not in writer_source for name in ("xdart", "PyQt", "PySide", "qtpy"))


def test_retained_v2_2d_writer_transposes_exactly_once(tmp_path):
    from xrd_tools.io.nexus import write_nexus_frame

    target = tmp_path / "retained.nexus"
    result = _r2(5)
    h5 = open_nexus_writer(target, overwrite=True)
    try:
        write_nexus_frame(h5, 0, result_2d=result)
    finally:
        h5.close()
    with h5py.File(target, "r") as h5:
        np.testing.assert_allclose(h5["entry/integrated_2d/intensity"][0], result.intensity.T)
