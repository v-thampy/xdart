from __future__ import annotations

import h5py
import numpy as np
import pytest

from xrd_tools.core import (
    FrameView,
    IntegrationResult1D,
    IntegrationResult2D,
    TwoDKind,
    assert_frameview_equivalent,
)
from xrd_tools.io import read_frame_view, read_frame_views, iter_frame_views
from xrd_tools.io.nexus import read_scan, write_integrated_stack


def _r1d(scale: float = 1.0) -> IntegrationResult1D:
    q = np.linspace(0.5, 5.0, 6)
    intensity = scale * np.linspace(10.0, 20.0, 6)
    return IntegrationResult1D(
        radial=q,
        intensity=intensity,
        sigma=np.sqrt(intensity),
        unit="q_A^-1",
    )


def _gi_2d(scale: float = 1.0) -> IntegrationResult2D:
    qip = np.linspace(-1.0, 2.0, 4)
    qoop = np.linspace(0.0, 3.0, 3)
    intensity = scale * np.arange(12, dtype=float).reshape(4, 3)
    return IntegrationResult2D(
        radial=qip,
        azimuthal=qoop,
        intensity=intensity,
        sigma=np.sqrt(intensity + 1.0),
        unit="qip_A^-1",
        azimuthal_unit="qoop_A^-1",
    )


def test_frame_view_from_results_uses_display_orientation():
    r2d = _gi_2d()

    view = FrameView.from_results(
        label=7,
        result_1d=_r1d(),
        result_2d=r2d,
        metadata_raw={"monitor": 5.0, "sample": "LaB6"},
        incident_angle=0.2,
    )

    assert view.two_d_kind is TwoDKind.QIP_QOOP
    assert view.axis_2d_x.unit == "qip_A^-1"
    assert view.axis_2d_y.unit == "qoop_A^-1"
    np.testing.assert_allclose(view.intensity_2d, r2d.intensity.T)
    np.testing.assert_allclose(view.sigma_2d, r2d.sigma.T)
    assert view.metadata_numeric == {"monitor": 5.0}


def test_write_read_frame_view_roundtrips_gi_2d_sigma_kind_and_metadata(tmp_path):
    path = tmp_path / "gi_frame_view.nxs"
    r1d = _r1d()
    r2d = _gi_2d()
    thumbnail = np.array([[0, 127], [255, 64]], dtype=np.uint8)

    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        write_integrated_stack(
            entry,
            frame_indices=[5],
            results_1d=[r1d],
            results_2d=[r2d],
        )
        geom = entry.create_group("per_frame_geometry")
        geom.create_dataset("frame_index", data=np.array([5], dtype=np.int64))
        geom.create_dataset("rot1", data=np.array([0.1], dtype=np.float32))
        geom.create_dataset("rot2", data=np.array([0.2], dtype=np.float32))
        geom.create_dataset("rot3", data=np.array([0.3], dtype=np.float32))
        geom.create_dataset("incident_angle", data=np.array([0.4], dtype=np.float32))
        scan_data = entry.create_group("scan_data")
        scan_data.create_dataset("frame_index", data=np.array([5], dtype=np.int64))
        scan_data.create_dataset("monitor", data=np.array([123.0], dtype=np.float32))
        frame = entry.create_group("frames/frame_0005")
        td = frame.create_dataset("thumbnail", data=thumbnail)
        td.attrs["vmin"] = 10.0
        td.attrs["vmax"] = 20.0
        td.attrs["dtype"] = "uint8"
        source = frame.create_group("source")
        source.create_dataset("path", data=np.bytes_("raw_master.h5"))
        source.create_dataset("frame_index", data=np.array(17, dtype=np.int64))

    ds = read_scan(path)
    assert ds["intensity_2d"].attrs["two_d_kind"] == TwoDKind.QIP_QOOP.value
    np.testing.assert_allclose(ds["sigma_2d"].values[0], r2d.sigma.T)

    loaded = read_frame_view(path, 5)
    expected = FrameView.from_results(
        label=5,
        result_1d=r1d,
        result_2d=r2d,
        thumbnail=10.0 + (thumbnail.astype(float) / 255.0) * 10.0,
        mask_baked=True,
        metadata_raw={"monitor": 123.0},
        incident_angle=0.4,
        source_path="raw_master.h5",
        source_frame_index=17,
    )
    assert_frameview_equivalent(expected, loaded)
    assert loaded.geometry is not None
    assert loaded.geometry.rot1 == np.float32(0.1)
    assert loaded.source_path == "raw_master.h5"
    assert loaded.source_frame_index == 17


def test_read_frame_views_reads_many_labels_with_one_contract(tmp_path):
    path = tmp_path / "many_frame_views.nxs"
    r1 = _r1d(1.0)
    r2 = _r1d(2.0)
    g1 = _gi_2d(1.0)
    g2 = _gi_2d(3.0)

    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        write_integrated_stack(
            entry,
            frame_indices=[5, 7],
            results_1d=[r1, r2],
            results_2d=[g1, g2],
        )
        scan_data = entry.create_group("scan_data")
        scan_data.create_dataset("frame_index", data=np.array([5, 7], dtype=np.int64))
        scan_data.create_dataset("monitor", data=np.array([10.0, 20.0], dtype=np.float32))

    views = read_frame_views(path)

    assert [view.label for view in views] == [5, 7]
    assert_frameview_equivalent(
        FrameView.from_results(
            label=5,
            result_1d=r1,
            result_2d=g1,
            metadata_raw={"monitor": 10.0},
        ),
        views[0],
    )
    assert_frameview_equivalent(
        FrameView.from_results(
            label=7,
            result_1d=r2,
            result_2d=g2,
            metadata_raw={"monitor": 20.0},
        ),
        views[1],
    )
    selected = read_frame_views(path, [7])
    assert len(selected) == 1
    assert_frameview_equivalent(views[1], selected[0])


def test_frame_view_reader_caches_scan_data_columns_per_open(tmp_path):
    # P2 #4 (perf): scan_data columns are read ONCE per open and sliced per
    # row, not re-read full for every (frame, column) — O(N^2) before.
    # Values must stay correct per frame.
    from xrd_tools.io.frame_view import FrameViewReader

    path = tmp_path / "cache_cols.nxs"
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        write_integrated_stack(
            entry, frame_indices=[5, 7], results_1d=[_r1d(1.0), _r1d(2.0)],
        )
        scan_data = entry.create_group("scan_data")
        scan_data.create_dataset("frame_index", data=np.array([5, 7], dtype=np.int64))
        scan_data.create_dataset("monitor", data=np.array([10.0, 20.0], dtype=np.float32))

    with FrameViewReader(path) as reader:
        assert reader._scan_data_columns is None              # not read yet
        assert reader.read(5).metadata_raw["monitor"] == 10.0
        cols = reader._scan_data_columns
        assert cols is not None and "monitor" in cols         # cached after 1st read
        # Second frame slices the SAME cached column array (no re-read) and
        # still gets its own row value.
        assert reader.read(7).metadata_raw["monitor"] == 20.0
        assert reader._scan_data_columns is cols


def test_iter_frame_views_streams_one_at_a_time(tmp_path):
    # P3 #6: iter_frame_views must yield frame-by-frame from one open reader
    # (a generator), not materialise the whole scan first.
    import types

    path = tmp_path / "stream.nxs"
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        write_integrated_stack(
            entry, frame_indices=[5, 7], results_1d=[_r1d(1.0), _r1d(2.0)],
        )
        scan_data = entry.create_group("scan_data")
        scan_data.create_dataset("frame_index", data=np.array([5, 7], dtype=np.int64))
        scan_data.create_dataset("monitor", data=np.array([10.0, 20.0], dtype=np.float32))

    gen = iter_frame_views(path)
    assert isinstance(gen, types.GeneratorType)        # lazy, not a list
    first = next(gen)                                  # one frame, rest unread
    assert first.label == 5
    assert [v.label for v in gen] == [7]               # remaining stream out

    # Eager wrapper yields the same set.
    assert [v.label for v in read_frame_views(path)] == [5, 7]


def test_frame_view_infers_gi_kind_for_old_files_without_explicit_attr(tmp_path):
    path = tmp_path / "old_gi_no_kind.nxs"
    r2d = _gi_2d()
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        write_integrated_stack(entry, frame_indices=[1], results_2d=[r2d])
        del entry["integrated_2d"].attrs["two_d_kind"]

    view = read_frame_view(path, 1)
    assert view.two_d_kind is TwoDKind.QIP_QOOP


def test_frame_view_equivalence_checks_label_and_numeric_metadata():
    base = FrameView.from_results(
        label=1,
        result_1d=_r1d(),
        metadata_raw={"monitor": 5.0, "sample": "LaB6"},
    )
    same = FrameView.from_results(
        label=1,
        result_1d=_r1d(),
        metadata_raw={"monitor": 5.0, "sample": "different text is ignored"},
    )
    assert_frameview_equivalent(base, same)

    different_label = FrameView.from_results(
        label=2,
        result_1d=_r1d(),
        metadata_raw={"monitor": 5.0},
    )
    with pytest.raises(AssertionError, match="label differs"):
        assert_frameview_equivalent(base, different_label)

    different_metadata = FrameView.from_results(
        label=1,
        result_1d=_r1d(),
        metadata_raw={"monitor": 6.0},
    )
    with pytest.raises(AssertionError, match="metadata_numeric"):
        assert_frameview_equivalent(base, different_metadata)
