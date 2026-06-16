# -*- coding: utf-8 -*-
"""Headless whole-scan aggregation (Step 7b core): xrd_tools.io.aggregate.

Covers the data_1d-replacement contract: aggregate over the COMPLETE on-disk
primary stack (the >64-frame coverage gate — the bug 7/8 exist to kill), the
on-disk ⊕ in-memory-tail combine (deduped by label), and NaN handling.
"""

from __future__ import annotations

import warnings

import h5py
import numpy as np
import pytest

from xrd_tools.io import aggregate_1d, aggregate_2d, get_1d

N_FRAMES = 100          # > the 64 store bound, on purpose (the Round-12 gate)
N_Q = 12
N_CHI = 8
LABELS = np.arange(1, N_FRAMES + 1, dtype=np.int32)   # 1-based


@pytest.fixture
def scan_file(tmp_path):
    p = tmp_path / "agg_100frame.nxs"
    q = np.linspace(0.5, 5.0, N_Q).astype(np.float32)
    q2 = np.linspace(0.5, 4.0, N_Q).astype(np.float32)
    chi = np.linspace(-180.0, 180.0, N_CHI, endpoint=False).astype(np.float32)
    # frame i (0-based) is filled with value (i+1) so the mean/sum over ALL
    # frames is exact and a subset would give a different answer.
    intensity_1d = np.tile(
        (np.arange(N_FRAMES) + 1.0)[:, None], (1, N_Q)).astype(np.float32)
    intensity_1d[:, 0] = np.nan                    # one all-NaN q-bin
    intensity_2d = np.tile(
        (np.arange(N_FRAMES) + 1.0)[:, None, None], (1, N_CHI, N_Q)).astype(np.float32)

    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        g1 = e.create_group("integrated_1d")
        g1.create_dataset("intensity", data=intensity_1d)
        qd = g1.create_dataset("q", data=q); qd.attrs["units"] = "1/angstrom"
        g1.create_dataset("frame_index", data=LABELS)
        g2 = e.create_group("integrated_2d")
        g2.create_dataset("intensity", data=intensity_2d)
        q2d = g2.create_dataset("q", data=q2); q2d.attrs["units"] = "1/angstrom"
        cd = g2.create_dataset("chi", data=chi); cd.attrs["units"] = "deg"
        g2.create_dataset("frame_index", data=LABELS)
    return p


def test_aggregate_1d_covers_all_frames_average_and_sum(scan_file):
    # The >64-frame gate: aggregate over the COMPLETE on-disk stack, not a subset.
    avg = aggregate_1d(scan_file, method="average")
    assert avg.n_frames == N_FRAMES
    np.testing.assert_allclose(avg.intensity[1:], np.mean(np.arange(1, N_FRAMES + 1)))  # 50.5
    assert np.isnan(avg.intensity[0])                       # all-NaN bin -> gap
    assert avg.q_unit == "1/angstrom"
    s = aggregate_1d(scan_file, method="sum")
    np.testing.assert_allclose(s.intensity[1:], np.sum(np.arange(1, N_FRAMES + 1)))  # 5050
    assert s.intensity[0] == 0.0                            # nansum of all-NaN -> 0


def test_aggregate_1d_matches_numpy_over_get_1d(scan_file):
    stack = get_1d(scan_file).intensity
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.nanmean(stack, axis=0)
    np.testing.assert_allclose(aggregate_1d(scan_file, method="average").intensity,
                               ref, equal_nan=True)


def test_aggregate_1d_with_in_memory_tail(scan_file):
    # The live case: frames not yet on disk are passed as `extra` and folded in.
    tail_labels = [N_FRAMES + 1, N_FRAMES + 2]
    tail = np.full((2, N_Q), 1000.0)
    s = aggregate_1d(scan_file, method="sum", extra=(tail_labels, tail))
    assert s.n_frames == N_FRAMES + 2
    np.testing.assert_allclose(s.intensity[1:], 5050.0 + 2000.0)   # disk + tail


def test_aggregate_1d_tail_dedups_by_label(scan_file):
    # A label present both on disk AND in the tail (freshly flushed yet resident)
    # is taken from the tail, never double-counted.
    overlap = np.full((1, N_Q), 7.0)
    s = aggregate_1d(scan_file, method="sum", extra=([50], overlap))
    assert s.n_frames == N_FRAMES                 # 100 disk - 1 overlap + 1 tail
    # disk sum minus the dropped row(50 -> value 50) plus the tail's 7
    np.testing.assert_allclose(s.intensity[1:], 5050.0 - 50.0 + 7.0)


def test_aggregate_2d_shape_and_values(scan_file):
    a = aggregate_2d(scan_file, method="average")
    assert a.intensity.shape == (N_CHI, N_Q)      # file/get_2d (n_chi, n_q) convention
    assert a.n_frames == N_FRAMES
    np.testing.assert_allclose(a.intensity, np.mean(np.arange(1, N_FRAMES + 1)))


def test_aggregate_average_does_not_warn_on_all_nan(scan_file):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Mean of empty slice")
        aggregate_1d(scan_file, method="average")  # the all-NaN bin must not warn


def test_aggregate_rejects_bad_method(scan_file):
    with pytest.raises(ValueError):
        aggregate_1d(scan_file, method="median")
