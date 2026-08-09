# -*- coding: utf-8 -*-
"""Mode-local durability for the production NeXus writer."""

from __future__ import annotations

import h5py
import numpy as np

NQ, NCHI = 16, 8
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


def _labels(nxs, group):
    with h5py.File(nxs, "r") as f:
        if group not in f["entry"]:
            return []
        return sorted(map(int, f[f"entry/{group}/frame_index"][()]))


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
