"""B-2 of NEXT-1210-VERIFY-ONCE: a finite Append that creates its output uses the fast finite writer.

The maintainer's picks (2026-11-11): "Fast; reprocess on failure (Recommended)" (a Stop keeps its verified frames; a
failed write discards that scan's output and the next Append reprocesses it) and "Fix in packet B (Recommended)"
(the fast failure path removes its own hidden partial file). Through the real executor and folder intent, as the
B-2 probe on the parent (``probes/b2_unfinished_append.py``) measured: 48 frames of one scan.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import pytest

import xrd_tools.io.output_transaction as transaction_api
from tests.xdart.scattering.test_equal_science_live_finite_reload import _exactly_equal
from tests.xdart.scattering.test_live_owner_transitions import _intent
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _terminals
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _start
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xrd_tools.core.frame_view import assert_frameview_equivalent
from xrd_tools.io import record_writer
from xrd_tools.io.append import _lineage_labels
from xrd_tools.io.frame_view import read_frame_view

FRAMES, STOP_AFTER = 48, 20
_COUNTED = ("_verify_dirty_evidence", "_seal_verified_stream_close", "_verify_fast_integrated_results")


def _raw(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in range(1, FRAMES + 1):
        _frame(raw, "scan", index, "txt", float(index))
    return raw


def _folder(tmp_path, raw, output, save, mode=None):
    intent = (_intent(tmp_path, raw, live=False, output=output) if mode is None
              else _intent(tmp_path, raw, live=False, mode=mode, output=output))
    intent.save_path = str(save)
    return intent


def _run(intent, *, stop_after=None):
    executor = StandardRunExecutor(join_timeout=5.0)
    identity, events, deadline = _start(executor, intent, request_value=7), [], time.monotonic() + 120
    try:
        while not any(event.kind in _TERMINAL for event in events):
            assert time.monotonic() < deadline, "no terminal"
            events.extend(executor.drain_events())
            ready = [event for event in events if event.kind is StandardEventKind.FRAME_READY]
            if stop_after is not None and len(ready) >= stop_after:
                executor.stop(identity)
                stop_after = None
            time.sleep(0.01)
    finally:
        executor.close(identity)
    return _terminals(events)


def _fail_second(monkeypatch, name):
    """The writer's second call of ``name`` raises once (the probe's injected failure)."""
    real, calls = getattr(record_writer.NexusRecordWriter, name), []

    def failing(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError(f"injected failure at the second {name}")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(record_writer.NexusRecordWriter, name, failing)
    return calls


def _counting(monkeypatch):
    counts = {name: 0 for name in _COUNTED}
    counts["whole_file_hashes"] = 0
    for name in _COUNTED:
        real = getattr(record_writer.NexusRecordWriter, name)

        def counted(self, *args, _real=real, _name=name, **kwargs):
            counts[_name] += 1
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(record_writer.NexusRecordWriter, name, counted)
    real_hash = transaction_api._sha256_handle

    def hashed(handle):
        counts["whole_file_hashes"] += 1
        return real_hash(handle)

    monkeypatch.setattr(transaction_api, "_sha256_handle", hashed)
    return counts


def _files(root):
    """The output folder's own files (the NeXus output and any hidden partial), not the scan's XYE folder."""
    return sorted(path.name for path in Path(root).iterdir() if path.is_file()) if Path(root).is_dir() else []


def _labels(path):
    with h5py.File(path, "r") as handle:
        return handle["entry/integrated_1d/frame_index"][()].tolist()


def _lineage(path):
    with h5py.File(path, "r") as handle:
        value = handle["entry/reduction/config/append_lineage"][()]
    return json.loads(value.decode() if isinstance(value, bytes) else str(value))


def test_a_fresh_append_counts_as_a_finite_overwrite(tmp_path, monkeypatch):
    raw = _raw(tmp_path)
    counts = _counting(monkeypatch)
    measured = {}
    for output in ("Overwrite", "Append"):
        for name in counts:
            counts[name] = 0
        save = tmp_path / output.lower()
        assert [status for status, _ in _run(_folder(tmp_path, raw, output, save))] == ["FINISHED"]
        measured[output] = dict(counts)
    # The parent: Append makes 2 checkpoint readbacks, 1 close reread and 2 whole-file hashes, and no fast readback.
    assert measured["Overwrite"]["_verify_fast_integrated_results"] == 1, measured
    assert measured["Append"] == measured["Overwrite"], measured


def test_a_stopped_fresh_append_keeps_its_frames_and_the_next_append_adds_the_rest(tmp_path):
    """Preservation (green on the parent): a Stop is a verified commit on either route."""
    raw = _raw(tmp_path)
    save = tmp_path / "append"
    first = _run(_folder(tmp_path, raw, "Append", save), stop_after=STOP_AFTER)
    assert [status for status, _ in first] == ["STOPPED"], first
    output = save / "scan_int1d.nexus"
    assert _files(save) == ["scan_int1d.nexus"]
    kept = _labels(output)
    assert kept == list(range(1, len(kept) + 1)) and STOP_AFTER <= len(kept) < FRAMES, kept
    lineage = _lineage(output)
    assert lineage["state"] == "committed" and list(_lineage_labels(lineage)) == kept, lineage
    second = _run(_folder(tmp_path, raw, "Append", save))
    assert second == [("FINISHED", f"Complete · {FRAMES} processed ({FRAMES - len(kept)} new) · 0 skipped · "
                                   f"0 pending · {FRAMES} discovered")], second
    assert _labels(output) == list(range(1, FRAMES + 1))
    reference = tmp_path / "overwrite"
    assert [status for status, _ in _run(_folder(tmp_path, raw, "Overwrite", reference))] == ["FINISHED"]
    for label in range(1, FRAMES + 1):  # B-E's exact comparison, on the resumed in-place output
        base, other = read_frame_view(reference / "scan_int1d.nexus", label), read_frame_view(output, label)
        for field in ("intensity_1d", "sigma_1d", "intensity_2d", "sigma_2d"):
            _exactly_equal(f"frame {label} {field}", getattr(base, field), getattr(other, field))
        assert (base.axis_1d.label, base.axis_1d.unit) == (other.axis_1d.label, other.axis_1d.unit)
        _exactly_equal(f"frame {label} axis_1d", base.axis_1d.values, other.axis_1d.values)
        assert dict(base.metadata_numeric) == dict(other.metadata_numeric)
        assert dict(base.metadata_raw) == dict(other.metadata_raw)
        assert_frameview_equivalent(base, other, rtol=0, atol=0)


@pytest.mark.parametrize("output", ("Append", "Overwrite"))
def test_a_failed_fresh_write_leaves_nothing_and_the_next_append_reprocesses_it(tmp_path, monkeypatch, output):
    """His picks: a failed write discards that scan's output, its own hidden partial included; the next Append
    reprocesses every frame. The writer's second batch write fails, so it stops partial mid-Run on both routes (this
    Run's flushes all fall in its finish, and a finish that fails once is retried and commits verified rows). The
    parent keeps a hidden partial on both routes."""
    raw = _raw(tmp_path)
    save = tmp_path / output.lower()
    _fail_second(monkeypatch, "_remember_written_rows")
    first = _run(_folder(tmp_path, raw, output, save))
    assert [status for status, _ in first] == ["FAILED"], first
    assert _files(save) == [], _files(save)
    monkeypatch.undo()
    second = _run(_folder(tmp_path, raw, "Append", save))
    assert second == [("FINISHED", f"Complete · {FRAMES} processed ({FRAMES} new) · 0 skipped · 0 pending · "
                                   f"{FRAMES} discovered")], second
    assert _files(save) == ["scan_int1d.nexus"] and _labels(save / "scan_int1d.nexus") == list(range(1, FRAMES + 1))


def test_the_fast_final_readback_refuses_a_changed_append_lineage(tmp_path, monkeypatch):
    """The final checkpoint clears the lineage's dirty bit before close; the fast final readback still compares the
    committed lineage (Codex's pre-writer review). On the parent a fresh Append never reaches that readback."""
    raw = _raw(tmp_path)
    save = tmp_path / "append"
    real = record_writer.NexusRecordWriter._verify_fast_integrated_results
    changed = []

    def corrupt_lineage_first(self, *args, **kwargs):
        # Every attempt: a finish retried after one refusal writes its lineage again, so a one-shot change is repaired.
        dataset = self._entry_group().get("reduction/config/append_lineage")
        if dataset is not None:
            lineage = json.loads(self._text_value(dataset[()]))
            lineage["epochs"] = []
            dataset[()] = json.dumps(lineage, sort_keys=True, separators=(",", ":"))
            changed.append(True)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(record_writer.NexusRecordWriter, "_verify_fast_integrated_results", corrupt_lineage_first)
    first = _run(_folder(tmp_path, raw, "Append", save))
    assert [status for status, _ in first] == ["FAILED"] and changed, (first, changed)
    assert "fast close changed its Append lineage" in first[0][1], first
    # A refused close holds its output under its public name, as a fast Overwrite's refused readback does today
    # (an integrity hold, not a failed write); that pre-existing hold is recorded, not changed, here.


def test_a_target_created_before_the_stream_reservation_is_never_clobbered(tmp_path, monkeypatch):
    """Preservation (green on the parent): the absent public name is reserved with O_CREAT|O_EXCL after admission."""
    raw = _raw(tmp_path)
    save = tmp_path / "append"
    target = save / "scan_int1d.nexus"
    foreign = b"another program's file, created after admission"
    real = transaction_api.OutputTransaction.begin_stream

    def intrude_then_begin(self, *args, **kwargs):
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(foreign)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(transaction_api.OutputTransaction, "begin_stream", intrude_then_begin)
    first = _run(_folder(tmp_path, raw, "Append", save))
    assert [status for status, _ in first] == ["FAILED"], first
    assert target.read_bytes() == foreign
    assert _files(save) == ["scan_int1d.nexus"], _files(save)


def test_a_fresh_live_append_keeps_the_resumable_writer(tmp_path, monkeypatch):
    """Live is excluded from B-2: a Live Run keeps one scan's session open across epochs, so its first epoch on a
    new output must stay resumable (not the fast writer, its empty seed or its no-partial rule). A Stop keeps the
    stopped epoch's frames."""
    from tests.xdart.scattering.test_live_qualify_refusal_run import _stop

    seen = {"admitted_fast": [], "writer_fast": []}
    real_admit = transaction_api.OutputTransactionCoordinator.admit

    def admit(self, *args, **kwargs):
        transaction = real_admit(self, *args, **kwargs)
        seen["admitted_fast"].append(transaction.fast_regenerable)
        return transaction

    real_readback = record_writer.NexusRecordWriter._verify_dirty_evidence

    def readback(self, *args, **kwargs):
        seen["writer_fast"].append(self._fast_regenerable)
        return real_readback(self, *args, **kwargs)

    monkeypatch.setattr(transaction_api.OutputTransactionCoordinator, "admit", admit)
    monkeypatch.setattr(record_writer.NexusRecordWriter, "_verify_dirty_evidence", readback)
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in range(1, 4):
        _frame(raw, "scan", index, "txt", float(index))
    intent = _intent(tmp_path, raw, live=True)
    intent.save_path = str(tmp_path / "live")
    executor = StandardRunExecutor(join_timeout=5.0)
    identity, events, deadline = _start(executor, intent, request_value=13), [], time.monotonic() + 120
    while len({event.frame_key.local_frame_label for event in events
               if event.kind is StandardEventKind.FRAME_READY}) < 3 and not any(
            event.kind in _TERMINAL for event in events):
        assert time.monotonic() < deadline, "frames never ready"
        events.extend(executor.drain_events())
        time.sleep(0.01)
    terminals = _stop(executor, identity, events)
    assert [status for status, _ in terminals] == ["STOPPED"], terminals
    assert seen["admitted_fast"] and not any(seen["admitted_fast"]), seen
    assert seen["writer_fast"] and not any(seen["writer_fast"]), seen
    (output,) = sorted((tmp_path / "live").rglob("*.nexus"))
    assert _labels(output) == [1, 2, 3]
