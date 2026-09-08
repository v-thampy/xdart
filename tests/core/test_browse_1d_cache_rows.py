from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.core import Axis
from xrd_tools.io import Browse1DCache, Frame1DModeRows, Frame1DRows, FrameScalarCatalog, FrameScalarRow, browse_1d_row_name


def _row(values):
    array = np.asarray(values, dtype=np.float64); array.setflags(write=False); return array


def _projection():
    axis, intensity, sigma = _row([.5, 1., 1.5]), _row([10., 11., 12.]), _row([1., 1.1, 1.2])
    catalog = FrameScalarCatalog("/processed/scan.nxs", "entry", (FrameScalarRow(2, modes_1d=("q",)),), axes_1d=(("q", "Q", "q_A^-1", False),))
    rows = Frame1DRows("/processed/scan.nxs", "entry", (2,), (Frame1DModeRows("q", Axis("Q", "q_A^-1", False, axis), (2,), (intensity,), (sigma,)),), "q")
    return catalog, rows, axis, intensity, sigma


def test_label_boundary_stores_complete_exact_named_rows() -> None:
    catalog, rows, axis, intensity, sigma = _projection(); cache = Browse1DCache(1024)
    keys = cache.store_1d_label(catalog, rows, 1, 2)
    assert tuple(key.name for key in keys) == tuple(browse_1d_row_name("q", name) for name in ("axis", "intensity", "sigma"))
    assert cache.resident_root_count == 3
    for key, expected in zip(keys, (axis, intensity, sigma), strict=True):
        with cache.borrow(key.frame, key.label, key.name) as borrowed: assert borrowed.array is expected
    cache.close()


def test_label_mismatch_refuses_before_cache_membership() -> None:
    catalog, rows, *_ = _projection(); cache = Browse1DCache(1024)
    wrong = Frame1DRows("/processed/other.nxs", rows.entry, rows.labels, rows.modes, rows.primary_mode)
    with pytest.raises(ValueError, match="different artifacts"):
        cache.store_1d_label(catalog, wrong, 1, 2)
    assert cache.resident_keys == ()
    cache.close()


def test_partial_label_repair_keeps_borrowed_survivor() -> None:
    catalog, rows, axis, *_ = _projection(); cache = Browse1DCache(1024)
    keys = cache.store_1d_label(catalog, rows, 1, 2)
    axis_key = next(key for key in keys if key.name.endswith(":axis")); borrowed = cache.borrow(axis_key.frame, axis_key.label, axis_key.name)
    cache.store_rows(9, 9, (("other", _row([3.])),))
    repaired = cache.store_1d_label(catalog, rows, 1, 2)
    assert repaired == keys and borrowed.array is axis
    borrowed.release(); cache.close()
