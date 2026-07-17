# -*- coding: utf-8 -*-
"""PF-1d outcome tests: Int 2D Append flush eligibility and per-mode durability.

The live-reproduced blocker: revisiting a 1D-only prefix, the FIRST periodic
2D flush saw an empty ``integrated_2d`` stack and widened its selection to
every label in the loaded union index; the still-uncomputed old labels went
into ``NexusWriteCursor.dropped`` and stayed excluded after their cakes were
computed — on disk, 2D held only the first save interval plus the new tail
(live file: 1D ``1..143``, 2D ``1..8`` + ``97..143``).

These tests drive the PRODUCTION save path (``LiveScan._save_to_nexus`` — the
exact call the wrangler's periodic flush makes) against real writer-produced
targets and assert ON-DISK row-set equality, per the correction contract:
more 1D-only rows than the save interval, multiple flushes, interrupt/resume,
a true no-op second Append, byte-unchanged pre-existing 1D, and a mutation
check proving the on-disk assertion detects the legacy behavior.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

NQ, NCHI = 16, 8
NOLD, NNEW = 25, 5           # 25 old 1D-only rows (> 2x the save interval)
SAVE_INTERVAL = 8            # matches production LIVE_SAVE_INTERVAL
RAW_H, RAW_W = 6, 5


def _result_1d(idx):
    from xrd_tools.core.containers import IntegrationResult1D
    return IntegrationResult1D(
        radial=np.linspace(0.5, 5.0, NQ, dtype=np.float32),
        intensity=np.full(NQ, float(idx + 1), dtype=np.float32),
        sigma=np.ones(NQ, dtype=np.float32),
        unit="q_A^-1",
    )


def _result_2d(idx):
    from xrd_tools.core.containers import IntegrationResult2D
    return IntegrationResult2D(
        radial=np.linspace(0.5, 5.0, NQ, dtype=np.float32),
        azimuthal=np.linspace(-10.0, 10.0, NCHI, dtype=np.float32),
        intensity=np.full((NQ, NCHI), float(idx + 1), dtype=np.float32),
        unit="q_A^-1",
    )


def _write_raw_stack(path, n):
    with h5py.File(path, "w") as f:
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        e.create_group("instrument/detector").create_dataset(
            "data",
            data=(np.arange(n * RAW_H * RAW_W, dtype=np.uint16)
                  .reshape(n, RAW_H, RAW_W)),
        )


def _frame_1d_only(idx, source_root):
    from xdart.modules.ewald.frame import LiveFrame
    fr = LiveFrame(idx=idx)
    fr.int_1d = _result_1d(idx)
    fr.scan_info = {"i0": float(idx + 1)}
    fr.source_file = "raw_stack.h5"
    fr.source_frame_idx = idx
    fr._source_root = str(source_root)
    fr.thumbnail = None          # source-only row: no stored thumbnail
    return fr


def _make_1d_only_target(tmp_path):
    """Phase 1: a real Int 1D run writes the 25-row 1D-only target."""
    from xdart.modules.ewald import LiveScan

    nxs = str(tmp_path / "scan.nxs")
    _write_raw_stack(tmp_path / "raw_stack.h5", NOLD + NNEW)
    scan = LiveScan(data_file=nxs)
    scan.skip_2d = True
    for i in range(NOLD):
        scan.add_frame(frame=_frame_1d_only(i, tmp_path), calculate=False,
                       update=True, get_sd=True, batch_save=True)
    scan._save_to_nexus()
    with h5py.File(nxs, "r") as f:
        assert sorted(map(int, f["entry/integrated_1d/frame_index"][()])) \
            == list(range(NOLD))
        assert "integrated_2d" not in f["entry"]
        baseline_1d = np.asarray(f["entry/integrated_1d/intensity"][()])
    return nxs, baseline_1d


def _labels(nxs, group):
    with h5py.File(nxs, "r") as f:
        if group not in f["entry"]:
            return []
        return sorted(map(int, f[f"entry/{group}/frame_index"][()]))


def _run_int2d_append(nxs, *, revisit, new, flush_every=SAVE_INTERVAL,
                      stop_after_flushes=None):
    """Drive the production Int 2D Append flush loop; returns the scan.

    ``revisit``: old labels to recompute 2D for; ``new``: labels appended with
    fresh 1D+2D+thumbnail.  Flushes via ``scan._save_to_nexus()`` every
    ``flush_every`` processed frames (the wrangler's periodic-save policy);
    stops mid-run after ``stop_after_flushes`` flushes when given.
    """
    from xdart.modules.ewald import LiveScan

    scan = LiveScan(data_file=nxs)
    scan.load_from_h5()
    scan.skip_2d = False
    mark = getattr(scan.frames, "mark_persisted", None)
    if callable(mark):
        mark(list(scan.frames.index))     # append target load marks loaded rows
    source_root = os.path.dirname(nxs)

    processed = 0
    flushes = 0
    for i in revisit:
        fr = scan.frames[i]
        fr.int_2d = _result_2d(i)
        scan.add_frame(frame=fr, calculate=False, update=True,
                       get_sd=True, batch_save=True)
        processed += 1
        if processed % flush_every == 0:
            scan._save_to_nexus()
            flushes += 1
            if stop_after_flushes is not None and flushes >= stop_after_flushes:
                return scan
    for i in new:
        fr = _frame_1d_only(i, source_root)
        fr.int_2d = _result_2d(i)
        fr.thumbnail = np.full((4, 4), float(i), dtype=np.float32)
        scan.add_frame(frame=fr, calculate=False, update=True,
                       get_sd=True, batch_save=True)
        processed += 1
        if processed % flush_every == 0:
            scan._save_to_nexus()
            flushes += 1
    scan._save_to_nexus(finalize=True)
    return scan


def test_int2d_append_backfills_full_prefix_across_flushes(tmp_path):
    """The headline PF-1d contract: every 1D-only frame gets its missing cake,
    existing 1D stays byte-identical, new frames get both, nothing is dropped."""
    nxs, baseline_1d = _make_1d_only_target(tmp_path)
    expected = list(range(NOLD + NNEW))

    scan = _run_int2d_append(
        nxs, revisit=range(NOLD), new=range(NOLD, NOLD + NNEW))

    assert _labels(nxs, "integrated_1d") == expected
    assert _labels(nxs, "integrated_2d") == expected, \
        "missing interior 2D prefix — the PF-1d data loss"
    cursor = getattr(scan, "_nexus_write_cursor", None)
    assert not (cursor and cursor.dropped), \
        "pending labels must never enter NexusWriteCursor.dropped"
    with h5py.File(nxs, "r") as f:
        # pre-existing 1D rows byte-unchanged (label-aligned prefix rows)
        final_1d = np.asarray(f["entry/integrated_1d/intensity"][()])
        idx_1d = [int(x) for x in f["entry/integrated_1d/frame_index"][()]]
        for label in range(NOLD):
            row = final_1d[idx_1d.index(label)]
            np.testing.assert_array_equal(row, baseline_1d[label])
        # valid 2D row shapes for every label
        shape_2d = f["entry/integrated_2d/intensity"].shape
        assert shape_2d == (NOLD + NNEW, NCHI, NQ)


def test_int2d_append_interrupt_then_resume_completes_exactly(tmp_path):
    """Interrupt after two flushes: exact durable 2D subset; resume writes
    only the remaining incomplete labels and reaches the full set."""
    nxs, _ = _make_1d_only_target(tmp_path)

    _run_int2d_append(nxs, revisit=range(NOLD), new=(),
                      stop_after_flushes=2)
    after_stop = _labels(nxs, "integrated_2d")
    assert after_stop == list(range(2 * SAVE_INTERVAL)), \
        "durable 2D subset after 2 flushes must be exactly the flushed labels"

    # Resume: the mode-aware completion cursor (PF-1c) revisits only the
    # incomplete labels.  Assert the resumed run wrote exactly the remainder.
    remaining = [i for i in range(NOLD) if i not in set(after_stop)]
    _run_int2d_append(nxs, revisit=remaining,
                      new=range(NOLD, NOLD + NNEW))
    assert _labels(nxs, "integrated_2d") == list(range(NOLD + NNEW))
    assert _labels(nxs, "integrated_1d") == list(range(NOLD + NNEW))


def test_second_int2d_append_is_true_noop(tmp_path):
    """After completion, another Append selects nothing and changes no bytes."""
    from xdart.modules.ewald import LiveScan
    from xdart.modules.ewald import nexus_writer

    nxs, _ = _make_1d_only_target(tmp_path)
    _run_int2d_append(nxs, revisit=range(NOLD), new=range(NOLD, NOLD + NNEW))

    def _snapshot():
        out = {}
        with h5py.File(nxs, "r") as f:
            for g in ("integrated_1d", "integrated_2d"):
                out[g] = (
                    np.asarray(f[f"entry/{g}/frame_index"][()]).tolist(),
                    np.asarray(f[f"entry/{g}/intensity"][()]).copy(),
                )
        return out

    before = _snapshot()
    scan = LiveScan(data_file=nxs)
    scan.load_from_h5()
    scan.skip_2d = False
    # the append selector must find NOTHING new for either group
    with h5py.File(nxs, "r") as f:
        for group in ("entry/integrated_1d", "entry/integrated_2d"):
            frames, _n = nexus_writer._new_frames_for_write(scan, f, group)
            assert frames == []
    scan._save_to_nexus(finalize=True)
    after = _snapshot()
    for g in before:
        assert after[g][0] == before[g][0]
        np.testing.assert_array_equal(after[g][1], before[g][1])


def test_pending_mode_label_is_not_marked_persisted(tmp_path):
    """Per-mode durability: a resident frame whose 2D result is still pending
    must not be marked persisted by a save that only committed its 1D row —
    a durable 1D row cannot authorize evicting a fresh 2D later."""
    from xdart.modules.ewald import LiveScan

    nxs = str(tmp_path / "scan.nxs")
    _write_raw_stack(tmp_path / "raw_stack.h5", 4)
    scan = LiveScan(data_file=nxs)
    scan.skip_2d = False                     # Int 2D run: 2D is an active mode
    for i in range(3):
        fr = _frame_1d_only(i, tmp_path)
        if i != 1:
            fr.int_2d = _result_2d(i)
        scan.add_frame(frame=fr, calculate=False, update=True,
                       get_sd=True, batch_save=True)
    scan._save_to_nexus()

    assert _labels(nxs, "integrated_1d") == [0, 1, 2]
    assert _labels(nxs, "integrated_2d") == [0, 2]
    persisted = set(getattr(scan.frames, "_persisted", set()))
    assert 0 in persisted and 2 in persisted
    assert 1 not in persisted, \
        "label with a pending 2D mode was marked persisted (evictable)"
    # ...and once its cake lands, the next flush writes it and marks it
    fr = scan.frames[1]
    fr.int_2d = _result_2d(1)
    scan.add_frame(frame=fr, calculate=False, update=True,
                   get_sd=True, batch_save=True)
    scan._save_to_nexus()
    assert _labels(nxs, "integrated_2d") == [0, 1, 2]
    assert 1 in set(getattr(scan.frames, "_persisted", set()))


def test_outcome_assertion_detects_legacy_flush_behavior(tmp_path, monkeypatch):
    """Mutation check (required by the correction contract): restore the
    121bc438 flush behavior — union-index widening plus pending-into-dropped —
    and require the on-disk row-set assertion to FAIL exactly as the live gate
    did (first interval + new tail only)."""
    from xdart.modules.ewald import nexus_writer

    real_select = nexus_writer._new_frames_for_write

    def legacy_new_frames_for_write(scan, h5f, group_path, cursor=None):
        # 121bc438: label-based selection over the WHOLE union index with no
        # residency filter — materializes and selects every not-on-disk label.
        existing_n = nexus_writer._existing_dataset_n(h5f, group_path)
        total_n = len(scan.frames.index)
        if existing_n > total_n:
            return [], -1
        on_disk = set()
        if group_path in h5f and "frame_index" in h5f[group_path]:
            on_disk = {int(x) for x in np.asarray(
                h5f[group_path]["frame_index"][()]).ravel()}
            live = {int(x) for x in scan.frames.index}
            if not on_disk.issubset(live):
                return [], -2
        drop = cursor.dropped.get(group_path, ()) if cursor else ()
        new_indices = [i for i in scan.frames.index
                       if int(i) not in on_disk and int(i) not in drop]
        return [scan.frames[i] for i in new_indices], existing_n

    real_prepare_2d = nexus_writer._prepare_integrated_2d

    def legacy_prepare_2d(f, scan, **kwargs):
        prepared = real_prepare_2d(f, scan, **kwargs)
        cursor = kwargs.get("cursor")
        # 121bc438: this save's pending labels became permanent drops.
        if cursor is not None and kwargs.get("replace_frame_indices") is None:
            for gp, labels in (cursor.pending or {}).items():
                cursor.dropped.setdefault(gp, set()).update(labels)
        return prepared

    monkeypatch.setattr(
        nexus_writer, "_new_frames_for_write", legacy_new_frames_for_write)
    monkeypatch.setattr(
        nexus_writer, "_prepare_integrated_2d", legacy_prepare_2d)

    nxs, _ = _make_1d_only_target(tmp_path)
    _run_int2d_append(nxs, revisit=range(NOLD), new=range(NOLD, NOLD + NNEW))

    on_disk_2d = _labels(nxs, "integrated_2d")
    assert on_disk_2d != list(range(NOLD + NNEW)), \
        "mutation not detected: the outcome assertion would pass under the " \
        "legacy behavior, so it cannot be trusted to pin the fix"
    # the legacy signature: first save interval + the new tail, interior missing
    assert on_disk_2d == (list(range(SAVE_INTERVAL))
                          + list(range(NOLD, NOLD + NNEW)))
    monkeypatch.setattr(nexus_writer, "_new_frames_for_write", real_select)
