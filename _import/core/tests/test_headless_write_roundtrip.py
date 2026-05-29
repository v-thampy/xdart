"""Round-trip: headless write path (write_nexus / write_nexus_frame /
open_nexus_writer) now produces the stacked v2 layout that read_scan
consumes — so reduce-headless → read-back works (the NexusSink schema
split fix)."""

from __future__ import annotations

import numpy as np

from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from ssrl_xrd_tools.io.nexus import (
    open_nexus_writer,
    read_scan,
    write_nexus,
    write_nexus_frame,
)

N_Q = 20
N_CHI = 8


def _r1d(seed: int) -> IntegrationResult1D:
    rng = np.random.default_rng(seed)
    return IntegrationResult1D(
        radial=np.linspace(0.5, 5.0, N_Q),
        intensity=rng.random(N_Q),
        sigma=rng.random(N_Q) * 0.1,
        unit="q_A^-1",
    )


def _r2d(seed: int) -> IntegrationResult2D:
    rng = np.random.default_rng(seed)
    return IntegrationResult2D(
        radial=np.linspace(0.5, 4.0, N_Q),
        azimuthal=np.linspace(-180, 180, N_CHI, endpoint=False),
        intensity=rng.random((N_Q, N_CHI)),   # (n_q, n_chi) per the convention
        unit="q_A^-1",
        azimuthal_unit="chi_deg",
    )


def test_write_nexus_batch_roundtrips_through_read_scan(tmp_path):
    p = tmp_path / "batch.nxs"
    r1 = {0: _r1d(0), 1: _r1d(1), 2: _r1d(2)}
    r2 = {0: _r2d(0), 1: _r2d(1), 2: _r2d(2)}
    write_nexus(p, results_1d=r1, results_2d=r2, overwrite=True)

    ds = read_scan(p)
    assert list(ds["frame"].values) == [0, 1, 2]
    assert ds["intensity_1d"].shape == (3, N_Q)
    assert ds["intensity_2d"].shape == (3, N_CHI, N_Q)   # (frame, chi, q)
    # values + orientation round-trip
    np.testing.assert_allclose(ds["intensity_1d"].values[1], r1[1].intensity, rtol=1e-6)
    np.testing.assert_allclose(
        ds["intensity_2d"].values[1], r2[1].intensity.T, rtol=1e-6,
    )
    assert ds["q"].attrs.get("units") == "q_A^-1"


def test_write_nexus_frame_incremental_roundtrips(tmp_path):
    """open_nexus_writer + write_nexus_frame (the NexusSink hot loop) must
    append rows that read_scan reads back in order."""
    p = tmp_path / "live.nxs"
    h5 = open_nexus_writer(p, overwrite=True)
    try:
        for i in (5, 6, 7):  # non-zero-based labels
            write_nexus_frame(h5, i, result_1d=_r1d(i), result_2d=_r2d(i))
        h5.flush()
    finally:
        h5.close()

    ds = read_scan(p)
    assert list(ds["frame"].values) == [5, 6, 7]
    assert ds["intensity_1d"].shape == (3, N_Q)
    assert ds["intensity_2d"].shape == (3, N_CHI, N_Q)
    np.testing.assert_allclose(ds["intensity_1d"].values[0], _r1d(5).intensity, rtol=1e-6)


def test_rewriting_a_frame_upserts_not_duplicates(tmp_path):
    """Writing the same frame label twice updates the row in place (no
    duplicate frame_index) — keeps reruns/partial reprocessing idempotent."""
    p = tmp_path / "rerun.nxs"
    h5 = open_nexus_writer(p, overwrite=True)
    try:
        write_nexus_frame(h5, 0, result_1d=_r1d(0), result_2d=_r2d(0))
        write_nexus_frame(h5, 1, result_1d=_r1d(1), result_2d=_r2d(1))
        # rerun frame 0 with fresh values
        new0 = IntegrationResult1D(
            radial=np.linspace(0.5, 5.0, N_Q), intensity=np.full(N_Q, 7.0),
            unit="q_A^-1",
        )
        write_nexus_frame(h5, 0, result_1d=new0)
    finally:
        h5.close()

    ds = read_scan(p, groups=("1d",))
    assert list(ds["frame"].values) == [0, 1]   # not [0, 1, 0]
    np.testing.assert_allclose(ds["intensity_1d"].values[0], 7.0)  # row 0 updated


def test_write_integrated_stack_bulk_then_incremental(tmp_path):
    """The shared stacked-write primitive: bulk-create on first save, then
    incremental upsert (re-saving a label replaces, new labels append)."""
    import h5py
    from ssrl_xrd_tools.io.nexus import write_integrated_stack

    p = tmp_path / "stack.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        # bulk create: 3 frames, 1D + 2D, compressed
        write_integrated_stack(
            e, frame_indices=[0, 1, 2],
            results_1d=[_r1d(0), _r1d(1), _r1d(2)],
            results_2d=[_r2d(0), _r2d(1), _r2d(2)],
            compression="lzf",
        )

    ds = read_scan(p)
    assert list(ds["frame"].values) == [0, 1, 2]
    assert ds["intensity_1d"].shape == (3, N_Q)
    assert ds["intensity_2d"].shape == (3, N_CHI, N_Q)

    # incremental: upsert frame 1, append frame 3
    new1 = IntegrationResult1D(
        radial=np.linspace(0.5, 5.0, N_Q), intensity=np.full(N_Q, 9.0), unit="q_A^-1",
    )
    with h5py.File(p, "a") as f:
        write_integrated_stack(
            f["entry"], frame_indices=[1, 3],
            results_1d=[new1, _r1d(3)], compression="lzf",
        )

    ds2 = read_scan(p, groups=("1d",))
    assert list(ds2["frame"].values) == [0, 1, 2, 3]      # 1 upserted, 3 appended
    np.testing.assert_allclose(ds2["intensity_1d"].values[1], 9.0)  # row 1 updated


def test_nexus_sink_roundtrips_through_read_scan(tmp_path):
    """The NexusSink itself (begin/write/finish) → read_scan."""
    import pytest
    pytest.importorskip("pyFAI")  # reduction pkg import chain needs it
    from ssrl_xrd_tools.reduction import NexusSink
    from ssrl_xrd_tools.reduction.core import FrameReduction, Frame, Scan

    p = tmp_path / "sink.nxs"
    sink = NexusSink(path=p, overwrite=True)
    scan = Scan(name="s", frames=[Frame(index=0, image=np.zeros((2, 2))),
                                  Frame(index=1, image=np.zeros((2, 2)))])
    sink.begin(scan, plan=None)
    for i in (0, 1):
        sink.write(
            Frame(index=i, image=np.zeros((2, 2))),
            FrameReduction(frame_index=i, result_1d=_r1d(i), result_2d=_r2d(i)),
        )
    sink.finish(result=None)

    ds = read_scan(p)
    assert list(ds["frame"].values) == [0, 1]
    assert ds["intensity_1d"].shape == (2, N_Q)
    assert ds["intensity_2d"].shape == (2, N_CHI, N_Q)
