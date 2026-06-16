# -*- coding: utf-8 -*-
"""H5FilePool pause/resume refcount (review_2026-06-15 §5).

A plain-set ``_paused`` let the FIRST resume reopen a file that a SECOND,
still-active writer had paused — a torn read mid-write once the 7+8 flush
fan-out makes overlapping pauses routine.  The Counter refcount keeps a path
paused until every pause is matched.
"""

from __future__ import annotations

import h5py
import numpy as np

from xdart.utils.h5pool import H5FilePool


def _make_file(tmp_path):
    p = tmp_path / "pool.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("x", data=np.arange(4))
    return str(p)


def test_pause_blocks_get_resume_unblocks(tmp_path):
    pool = H5FilePool(max_open=2)
    path = _make_file(tmp_path)
    assert pool.get(path) is not None
    pool.pause(path)
    assert pool.get(path) is None        # paused: must not reopen
    pool.resume(path)
    assert pool.get(path) is not None     # un-paused
    pool.close_all()


def test_nested_pause_needs_matching_resumes(tmp_path):
    # The §5 regression: two concurrent writers pause the same file.
    pool = H5FilePool(max_open=2)
    path = _make_file(tmp_path)
    pool.pause(path)                      # writer A
    pool.pause(path)                      # writer B
    pool.resume(path)                     # A done — B still writing
    assert pool.get(path) is None         # MUST stay paused (the bug = not None)
    pool.resume(path)                     # B done
    assert pool.get(path) is not None     # now safe to reopen
    pool.close_all()


def test_unbalanced_resume_is_safe(tmp_path):
    # A stray resume on a never-paused path must not drive the count negative
    # (which would make a later single pause fail to block).
    pool = H5FilePool(max_open=2)
    path = _make_file(tmp_path)
    pool.resume(path)                     # never paused
    pool.resume(path)
    pool.pause(path)
    assert pool.get(path) is None         # one pause still blocks
    pool.resume(path)
    assert pool.get(path) is not None
    pool.close_all()
