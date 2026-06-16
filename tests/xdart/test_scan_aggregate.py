# -*- coding: utf-8 -*-
"""Whole-scan aggregation over a live scan (greenfield Step 7b, xdart half).

The Round-12 regression gate (review_2026-06-15 §2.E): a whole-scan Sum/Average
must cover EVERY frame even when the scan is longer than the in-memory cap — the
frames live as an on-disk prefix ⊕ an unflushed in-memory tail, never all in RAM.
These tests build a REAL LiveScan whose frames are genuinely split across disk
and memory (cap << N, periodic saves, no final save) and assert the aggregate ==
the analytic reference over all N, plus normalization parity and the
primary-mode-scoped refusal predicate.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from xdart.modules.scan_aggregate import (
    mode_aggregation_allowed,
    whole_scan_aggregate_1d,
    whole_scan_aggregate_2d,
)

NQ, NCHI = 6, 4
CAP = 8
N = 95                        # >> CAP, and >> the store's 64 heavy bound
_Q = np.linspace(0.5, 5.0, NQ, dtype=np.float32)
_CHI = np.linspace(-90.0, 90.0, NCHI, dtype=np.float32)


def _frame(idx, *, with_2d):
    from xdart.modules.ewald.frame import LiveFrame
    from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
    fr = LiveFrame(idx=idx)
    fr.int_1d = IntegrationResult1D(
        radial=_Q, intensity=np.full(NQ, float(idx + 1), np.float32),
        sigma=np.ones(NQ, np.float32), unit="q_A^-1")
    if with_2d:
        fr.int_2d = IntegrationResult2D(            # (radial, azimuthal) = (nq, nchi)
            radial=_Q, azimuthal=_CHI,
            intensity=np.full((NQ, NCHI), float(idx + 1), np.float32),
            unit="q_A^-1", azimuthal_unit="chi_deg")
    fr.scan_info = {"i0": float(idx + 1)}
    fr.source_file = ""
    fr.source_frame_idx = 0
    fr.skip_map_raw = True
    return fr


def _split_scan(tmp_path, *, with_2d):
    """A LiveScan with frames split across disk (0..89) and an unflushed
    in-memory tail (90..94): cap=8 forces FIFO eviction of persisted frames, and
    saving every 10 with NO final save leaves the last 5 only in memory."""
    from xdart.modules.ewald import LiveScan
    from xrd_tools.io import get_1d
    scan = LiveScan(data_file=str(tmp_path / "scan.nxs"))
    scan.skip_2d = not with_2d
    scan.frames._in_memory_cap = CAP
    for i in range(N):
        scan.add_frame(frame=_frame(i, with_2d=with_2d), calculate=False,
                       update=True, get_sd=True, batch_save=True)
        if (i + 1) % 10 == 0:                  # saves after 9,19,...,89; none after
            scan._save_to_nexus()
    # Prove the split the test depends on: disk holds the prefix, memory the tail.
    on_disk = set(int(x) for x in get_1d(scan.data_file).frames)
    unflushed = {int(idx) for idx in scan.frames._in_memory
                 if idx not in scan.frames._persisted}
    assert on_disk == set(range(90)), sorted(on_disk)
    assert unflushed == set(range(90, 95)), sorted(unflushed)
    assert len(scan.frames._in_memory) < N      # NOT all in RAM — disk read needed
    return scan


def test_whole_scan_1d_covers_disk_prefix_and_memory_tail(tmp_path):
    scan = _split_scan(tmp_path, with_2d=False)
    avg = whole_scan_aggregate_1d(scan, method="average")
    assert avg is not None and avg.n_frames == N
    np.testing.assert_allclose(avg.intensity, np.mean(np.arange(1, N + 1)))   # 48.0
    s = whole_scan_aggregate_1d(scan, method="sum")
    np.testing.assert_allclose(s.intensity, float(np.sum(np.arange(1, N + 1))))  # 4560
    assert s.q_unit and "q" in str(s.q_unit).lower()    # unit carried through
    np.testing.assert_allclose(np.asarray(s.q), _Q)     # q axis carried through


def test_whole_scan_2d_covers_all_and_uses_disk_orientation(tmp_path):
    scan = _split_scan(tmp_path, with_2d=True)
    avg = whole_scan_aggregate_2d(scan, method="average")
    assert avg is not None and avg.n_frames == N
    assert avg.intensity.shape == (NCHI, NQ)          # disk/get_2d (n_chi, n_q)
    np.testing.assert_allclose(avg.intensity, np.mean(np.arange(1, N + 1)))


@pytest.mark.parametrize("dim, func, with_2d", [
    ("1d", whole_scan_aggregate_1d, False),
    ("2d", whole_scan_aggregate_2d, True),
])
def test_whole_scan_aggregate_waits_for_shared_file_lock(
    tmp_path, dim, func, with_2d,
):
    scan = _split_scan(tmp_path, with_2d=with_2d)
    result = {}
    error = {}

    scan.file_lock.acquire()
    try:
        def _read():
            try:
                result["value"] = func(scan, method="average")
            except Exception as exc:  # pragma: no cover - diagnostic only
                error["exc"] = exc

        thread = threading.Thread(target=_read, name=f"aggregate-{dim}")
        thread.start()
        time.sleep(0.1)
        assert thread.is_alive()
        assert result == {}
        assert error == {}
    finally:
        scan.file_lock.release()

    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert error == {}
    assert result["value"] is not None
    assert result["value"].n_frames == N


def test_whole_scan_aggregate_dedups_tail_flushed_after_snapshot(
    tmp_path, monkeypatch,
):
    # Simulates the interleave where _unflushed_tail snapshots label 50, then
    # the writer flushes it before aggregate_1d reads the file.  The wrapper
    # must rely on xrd_tools.io.aggregate's label dedupe and count that label
    # once, not disk+tail.
    from xdart.modules import scan_aggregate as scan_aggregate_mod

    scan = _split_scan(tmp_path, with_2d=False)
    overlap = _frame(50, with_2d=False)
    overlap.int_1d.intensity = np.full(NQ, 7.0, np.float32)
    monkeypatch.setattr(
        scan_aggregate_mod, "_unflushed_tail", lambda _scan: [(50, overlap)],
    )

    summed = scan_aggregate_mod.whole_scan_aggregate_1d(scan, method="sum")

    assert summed.n_frames == 90
    disk_prefix_sum = float(np.sum(np.arange(1, 91)))
    np.testing.assert_allclose(summed.intensity, disk_prefix_sum - 51.0 + 7.0)


def test_whole_scan_1d_normalizes_before_reducing(tmp_path):
    # End-to-end §2.B parity through the xdart wrapper: divisor == the frame's own
    # value -> every normalized frame is 1.0, so average==1.0 / sum==N.
    scan = _split_scan(tmp_path, with_2d=False)
    norm = {i: float(i + 1) for i in range(N)}
    avg = whole_scan_aggregate_1d(scan, method="average", norm=norm)
    np.testing.assert_allclose(avg.intensity, 1.0)
    s = whole_scan_aggregate_1d(scan, method="sum", norm=norm)
    np.testing.assert_allclose(s.intensity, float(N))


def test_whole_scan_1d_norm_channel_reads_scan_data_and_tail(tmp_path):
    # §2.B end-to-end via the norm_channel path: the divisor for each frame is
    # read from the file's scan_data (disk prefix) AND the tail frames' scan_info
    # (i0 == frame value), so every normalized frame is 1.0 across the split.
    scan = _split_scan(tmp_path, with_2d=False)
    avg = whole_scan_aggregate_1d(scan, method="average", norm_channel="i0")
    np.testing.assert_allclose(avg.intensity, 1.0)
    s = whole_scan_aggregate_1d(scan, method="sum", norm_channel="i0")
    np.testing.assert_allclose(s.intensity, float(N))
    # An absent channel = no normalization (parity with legacy's `if value>0`).
    raw = whole_scan_aggregate_1d(scan, method="average", norm_channel="not_a_channel")
    np.testing.assert_allclose(raw.intensity, np.mean(np.arange(1, N + 1)))


def test_whole_scan_defers_when_nothing_on_disk(tmp_path):
    from xdart.modules.ewald import LiveScan
    scan = LiveScan(data_file=str(tmp_path / "fresh.nxs"))
    scan.skip_2d = True
    for i in range(3):                              # added, never saved
        scan.add_frame(frame=_frame(i, with_2d=False), calculate=False,
                       update=True, get_sd=True, batch_save=True)
    assert whole_scan_aggregate_1d(scan, method="average") is None


@pytest.mark.parametrize("displayed, primary, allowed", [
    ("default", "default", True),
    (None, None, True),
    (None, "default", True),
    ("q_total", "q_total", True),
    ("q_ip", "default", False),       # non-primary GI sub-mode -> must refuse
    ("q_ip", "q_oop", False),
])
def test_mode_aggregation_allowed(displayed, primary, allowed):
    assert mode_aggregation_allowed(displayed, primary) is allowed
