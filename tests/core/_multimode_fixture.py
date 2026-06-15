"""Deterministic multi-mode GI scan -> .nxs via ``write_frame_records``.

Pure-io (no xdart): used to CAPTURE the per-mode reference signature and by the
gate test to re-write the identical scan.  Locks the NEW nested-subgroup layout
(subgroup names, primary_mode/multi_result_modes attrs, per-mode stacks) so any
accidental change to the frozen-once-shipped multi-result format fails loud.
"""
from __future__ import annotations

import h5py
import numpy as np

from xrd_tools.core import FrameRecord, FrameView, TwoDKind, axis_from_unit
from xrd_tools.io import write_frame_records


def _view(label, scale, *, nq, nchi, x1, x2, y2, kind):
    return FrameView(
        label=label,
        axis_1d=axis_from_unit(x1, np.linspace(1.0, 2.0, nq)),
        intensity_1d=(np.arange(nq, dtype=np.float32) * scale + label),
        axis_2d_x=axis_from_unit(x2, np.linspace(0.0, 1.0, nq)),
        axis_2d_y=axis_from_unit(y2, np.linspace(0.0, 1.0, nchi)),
        intensity_2d=(np.arange(nchi * nq, dtype=np.float32).reshape(nchi, nq)
                      * scale + label),
        two_d_kind=kind,
    )


def write_reference_multimode_scan(out_path):
    """3-frame GI scan: 1D modes {q_total(primary), q_oop}, 2D modes
    {qip_qoop(primary), q_chi}.  Distinct bin counts per mode."""
    recs = []
    for fi in range(3):
        r = FrameRecord.from_view(
            _view(fi, 1.0, nq=5, nchi=3, x1="q_A^-1", x2="qip_A^-1",
                  y2="qoop_A^-1", kind=TwoDKind.QIP_QOOP),
            mode_1d="q_total", mode_2d="qip_qoop",
        )
        r = r.with_result_1d(
            "q_oop",
            _view(fi, 9.0, nq=7, nchi=3, x1="qoop_A^-1", x2="qip_A^-1",
                  y2="qoop_A^-1", kind=TwoDKind.QIP_QOOP),
            make_active=False,
        )
        r = r.with_result_2d(
            "q_chi",
            _view(fi, 5.0, nq=5, nchi=4, x1="q_A^-1", x2="q_A^-1",
                  y2="chi_deg", kind=TwoDKind.Q_CHI),
            make_active=False,
        )
        recs.append(r)
    with h5py.File(out_path, "w") as f:
        write_frame_records(f.create_group("entry"), recs)
    return out_path
