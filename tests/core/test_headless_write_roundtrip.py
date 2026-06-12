"""Round-trip: headless write path (write_nexus / write_nexus_frame /
open_nexus_writer) now produces the stacked v2 layout that read_scan
consumes — so reduce-headless → read-back works (the NexusSink schema
split fix)."""

from __future__ import annotations

import numpy as np

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.io.nexus import (
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
    from xrd_tools.io.nexus import write_integrated_stack

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


def test_monotonic_append_fast_path_falls_back_after_late_frame(tmp_path):
    import h5py
    from xrd_tools.io.nexus import write_integrated_stack

    p = tmp_path / "late_frame.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        write_integrated_stack(e, frame_indices=[0, 2],
                               results_1d=[_r1d(0), _r1d(2)])
        assert bool(e["integrated_1d"].attrs["_frame_index_strictly_increasing"])
        write_integrated_stack(e, frame_indices=[1], results_1d=[_r1d(1)])
        assert not bool(e["integrated_1d"].attrs["_frame_index_strictly_increasing"])
        write_integrated_stack(e, frame_indices=[2], results_1d=[_r1d(7)])
        np.testing.assert_array_equal(e["integrated_1d/frame_index"][()], [0, 2, 1])


def test_write_stitched_roundtrips_through_read_stitched(tmp_path):
    """write_stitched ↔ read_stitched: stitched_2d is (q, chi) as-is (NOT
    transposed like integrated_2d)."""
    import h5py
    from xrd_tools.io.nexus import write_stitched, read_stitched

    p = tmp_path / "stitched.nxs"
    s1 = _r1d(0)                       # (N_Q,)
    s2 = IntegrationResult2D(          # intensity (N_Q, N_CHI) — q-major, as-is
        radial=np.linspace(0.5, 4.0, N_Q),
        azimuthal=np.linspace(-180, 180, N_CHI, endpoint=False),
        intensity=np.random.default_rng(3).random((N_Q, N_CHI)),
        unit="q_A^-1", azimuthal_unit="chi_deg",
    )
    with h5py.File(p, "w") as f:
        write_stitched(f.create_group("entry"), stitched_1d=s1, stitched_2d=s2)

    ds = read_stitched(p)
    assert ds["stitched_1d"].dims == ("q",)
    assert ds["stitched_2d"].dims == ("q", "chi")
    assert ds["stitched_2d"].shape == (N_Q, N_CHI)
    np.testing.assert_allclose(ds["stitched_2d"].values, s2.intensity, rtol=1e-6)
    np.testing.assert_allclose(ds["stitched_1d"].values, s1.intensity, rtol=1e-6)


def test_write_integrated_stack_shape_change_rewrites(tmp_path):
    """Reintegration at a different npt (C3): the group is rewritten with the
    new row size + refreshed q axis, not a slice-assign that would raise."""
    import h5py
    from xrd_tools.io.nexus import write_integrated_stack

    p = tmp_path / "reint.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        write_integrated_stack(e, frame_indices=[0, 1, 2],
                               results_1d=[_r1d(0), _r1d(1), _r1d(2)])  # npt=N_Q
    # reintegrate all frames at a different npt
    big = lambda s: IntegrationResult1D(
        radial=np.linspace(0.5, 5.0, N_Q * 2),
        intensity=np.full(N_Q * 2, float(s)), unit="q_A^-1",
    )
    with h5py.File(p, "a") as f:
        write_integrated_stack(f["entry"], frame_indices=[0, 1, 2],
                               results_1d=[big(0), big(1), big(2)])

    ds = read_scan(p, groups=("1d",))
    assert ds["intensity_1d"].shape == (3, N_Q * 2)   # rewritten at new npt
    assert ds["q"].shape == (N_Q * 2,)                # q axis refreshed
    assert list(ds["frame"].values) == [0, 1, 2]


def test_write_integrated_stack_partial_shape_change_raises(tmp_path):
    """P1: a shape change (different npt) on a batch that does NOT cover
    every on-disk frame must raise, not silently drop the omitted rows."""
    import h5py
    import pytest
    from xrd_tools.io.nexus import write_integrated_stack

    p = tmp_path / "partial_reint.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        write_integrated_stack(e, frame_indices=[0, 1, 2],
                               results_1d=[_r1d(0), _r1d(1), _r1d(2)])  # npt=N_Q
    big = lambda s: IntegrationResult1D(
        radial=np.linspace(0.5, 5.0, N_Q * 2),
        intensity=np.full(N_Q * 2, float(s)), unit="q_A^-1",
    )
    # Rewrite only frame 1 at a new npt — would drop 0 and 2.
    with h5py.File(p, "a") as f:
        with pytest.raises(ValueError, match="missing frame"):
            write_integrated_stack(f["entry"], frame_indices=[1],
                                   results_1d=[big(1)])
    # Original three frames are still intact (the failed call didn't delete).
    ds = read_scan(p, groups=("1d",))
    assert list(ds["frame"].values) == [0, 1, 2]
    assert ds["intensity_1d"].shape == (3, N_Q)


def test_write_integrated_stack_rejects_duplicate_labels(tmp_path):
    import h5py
    import pytest
    from xrd_tools.io.nexus import write_integrated_stack
    p = tmp_path / "dup.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        with pytest.raises(ValueError, match="duplicate"):
            write_integrated_stack(
                e, frame_indices=[0, 0], results_1d=[_r1d(0), _r1d(1)],
            )


def test_write_nexus_validates_complete_batch_before_creating_file(tmp_path):
    """A divergent later row must fail before the output is mutated."""
    import pytest

    p = tmp_path / "atomic_batch.nxs"
    bad = IntegrationResult1D(
        radial=np.linspace(0.5, 6.0, N_Q),
        intensity=np.ones(N_Q),
        unit="q_A^-1",
    )
    with pytest.raises(ValueError, match="radial axis"):
        write_nexus(p, results_1d={0: _r1d(0), 1: bad}, overwrite=True)
    assert not p.exists()


def test_write_nexus_rejects_normalized_duplicate_labels(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="duplicate normalized"):
        write_nexus(
            tmp_path / "duplicate_labels.nxs",
            results_1d={1: _r1d(0), "1": _r1d(1)},
            overwrite=True,
        )


def test_write_nexus_preflights_existing_1d_2d_before_mutation(tmp_path):
    """If 2D is incompatible, a compatible 1D tail must not be appended."""
    import h5py
    import pytest

    p = tmp_path / "no_half_commit.nxs"
    write_nexus(
        p,
        results_1d={0: _r1d(0)},
        results_2d={0: _r2d(0)},
        overwrite=True,
    )
    bad_2d = _r2d(1)
    bad_2d.radial = np.linspace(9.0, 10.0, N_Q)
    with pytest.raises(ValueError, match="integrated_2d"):
        write_nexus(
            p,
            results_1d={1: _r1d(1)},
            results_2d={1: bad_2d},
        )
    with h5py.File(p, "r") as f:
        np.testing.assert_array_equal(f["entry/integrated_1d/frame_index"][()], [0])
        np.testing.assert_array_equal(f["entry/integrated_2d/frame_index"][()], [0])


def test_nexus_sink_roundtrips_through_read_scan(tmp_path):
    """The NexusSink itself (begin/write/finish) → read_scan."""
    import pytest
    pytest.importorskip("pyFAI")  # reduction pkg import chain needs it
    from xrd_tools.reduction import NexusSink
    from xrd_tools.reduction.core import FrameReduction, Frame, Scan

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


def test_write_integrated_stack_axis_change_same_bincount_rewrites(tmp_path):
    """P1: a reintegration that keeps the bin count but changes the unit/
    axis (q_A^-1 → 2th_deg) must refresh the stored q axis + units, not
    just the intensity rows."""
    import h5py
    from xrd_tools.io.nexus import write_integrated_stack

    p = tmp_path / "axis_change.nxs"
    q = np.linspace(0.5, 5.0, N_Q)
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        write_integrated_stack(
            e, frame_indices=[0, 1, 2],
            results_1d=[IntegrationResult1D(radial=q, intensity=np.full(N_Q, s),
                                            unit="q_A^-1") for s in (0, 1, 2)],
        )
    tth = np.linspace(5.0, 45.0, N_Q)   # same bin count, different axis+unit
    with h5py.File(p, "a") as f:
        write_integrated_stack(
            f["entry"], frame_indices=[0, 1, 2],
            results_1d=[IntegrationResult1D(radial=tth, intensity=np.full(N_Q, s),
                                            unit="2th_deg") for s in (0, 1, 2)],
        )
    ds = read_scan(p, groups=("1d",))
    assert ds["q"].attrs.get("units") == "2th_deg"        # unit refreshed
    np.testing.assert_allclose(ds["q"].values[-1], 45.0, atol=1e-3)  # axis refreshed


def test_write_integrated_stack_sigma_stays_row_aligned(tmp_path):
    """P1: once a sigma dataset exists, every appended frame extends it
    (NaN-padded when that frame has no sigma) so intensity never outgrows
    sigma — which would make read_scan raise on the shape mismatch."""
    import h5py
    from xrd_tools.io.nexus import write_integrated_stack

    p = tmp_path / "sigma_align.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        write_integrated_stack(
            e, frame_indices=[0, 1],
            results_1d=[_r1d(0), _r1d(1)],   # _r1d carries sigma
        )
    # Append a frame WITHOUT sigma.
    no_sigma = IntegrationResult1D(
        radial=np.linspace(0.5, 5.0, N_Q), intensity=np.full(N_Q, 9.0),
        unit="q_A^-1",
    )
    with h5py.File(p, "a") as f:
        write_integrated_stack(f["entry"], frame_indices=[2], results_1d=[no_sigma])
    with h5py.File(p, "r") as f:
        assert (f["entry/integrated_1d/intensity"].shape
                == f["entry/integrated_1d/sigma"].shape == (3, N_Q))
    ds = read_scan(p, groups=("1d",))   # must not raise
    assert np.all(np.isnan(ds["sigma_1d"].values[2]))   # padded row is NaN
