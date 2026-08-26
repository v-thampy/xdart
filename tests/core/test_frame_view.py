from __future__ import annotations

import sys

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


@pytest.mark.parametrize(
    ("case", "node_name", "message"),
    (
        ("inventory", "/entry/integrated_1d/frame_index", "item count"),
        ("axis", "/entry/integrated_1d/q", "item count"),
        ("scan_data", "/entry/scan_data/oversized", "columns exceed"),
        ("row_1d", "/entry/integrated_1d/intensity", "1-D row stack"),
        ("row_2d", "/entry/integrated_2d/intensity", "2-D row stack"),
    ),
)
def test_frame_view_refuses_malformed_or_oversized_nodes_before_getitem(
    tmp_path, monkeypatch, case, node_name, message,
):
    """Sparse foreign shapes are rejected from HDF metadata, before a read."""

    from xrd_tools.io.frame_view import FrameViewReader

    path = tmp_path / f"bounded_{case}.nxs"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        write_integrated_stack(
            entry,
            frame_indices=[5],
            results_1d=[_r1d()],
            results_2d=[_gi_2d()],
        )
        scan_data = entry.create_group("scan_data")
        scan_data.create_dataset("frame_index", data=np.array([5], dtype=np.int64))
        scan_data.create_dataset("monitor", data=np.array([1.0], dtype=np.float32))
        if case == "inventory":
            group = entry["integrated_1d"]
            del group["frame_index"]
            group.create_dataset(
                "frame_index", shape=(1_000_001,), dtype=np.int64, chunks=(64,),
            )
        elif case == "axis":
            group = entry["integrated_1d"]
            del group["q"]
            group.create_dataset("q", shape=(1_000_001,), dtype=np.float64, chunks=(64,))
        elif case == "scan_data":
            scan_data.create_dataset(
                "oversized", shape=(1,), dtype="S67108865", chunks=(1,),
            )
        elif case == "row_1d":
            group = entry["integrated_1d"]
            del group["intensity"]
            group.create_dataset("intensity", shape=(1, 7), dtype=np.float64)
        else:
            group = entry["integrated_2d"]
            del group["intensity"]
            group.create_dataset("intensity", shape=(1, 4, 4), dtype=np.float64)

    real_getitem = h5py.Dataset.__getitem__
    probed = []

    def guarded(dataset, selection):
        if dataset.name == node_name:
            probed.append(selection)
            raise AssertionError(f"oversized node was read: {node_name}")
        return real_getitem(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    with pytest.raises(ValueError, match=message):
        with FrameViewReader(path):
            pass
    assert probed == []


def test_frame_view_bounds_source_path_and_text_attributes_without_getitem(
    tmp_path, monkeypatch,
):
    from xrd_tools.io.frame_view import FrameViewReader

    path = tmp_path / "bounded_text.nxs"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        write_integrated_stack(
            entry, frame_indices=[5], results_1d=[_r1d()],
        )
        frame = entry.create_group("frames/frame_0005/source")
        frame.create_dataset(
            "path", data="x" * 4097, dtype=h5py.string_dtype("utf-8"),
        )
        frame.create_dataset("frame_index", data=np.int64(0))

    real_getitem = h5py.Dataset.__getitem__
    probed = []

    def guarded(dataset, selection):
        if dataset.name.endswith("/source/path"):
            probed.append(selection)
            raise AssertionError("source path used Dataset.__getitem__")
        return real_getitem(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    with FrameViewReader(path) as reader:
        with pytest.raises(ValueError, match="UTF-8 byte ceiling"):
            reader.read(5)
    assert probed == []

    with h5py.File(path, "r+") as handle:
        del handle["entry/frames/frame_0005/source/path"]
        handle["entry/integrated_1d/q"].attrs["units"] = "u" * 257
    with pytest.raises(ValueError, match="text byte ceiling"):
        with FrameViewReader(path):
            pass


def test_frame_view_bounds_source_base_without_attribute_getitem_or_file_name(
    tmp_path, monkeypatch,
):
    from xrd_tools.io.frame_view import FrameViewReader

    oversized = tmp_path / "oversized_source_base.nxs"
    ordinary = tmp_path / "ordinary_source_base.nxs"
    for path, source_base in (
        (oversized, "x" * 4097),
        (ordinary, "/relocated/raw"),
    ):
        with h5py.File(path, "w") as handle:
            handle.attrs["file_name"] = "must-not-be-read" * 5000
            entry = handle.create_group("entry")
            entry.attrs["source_base"] = source_base
            write_integrated_stack(
                entry, frame_indices=[5], results_1d=[_r1d()],
            )

    real_getitem = h5py.AttributeManager.__getitem__
    probed = []

    def guarded(attrs, name):
        if name in {"source_base", "file_name"}:
            probed.append(name)
            raise AssertionError(f"attribute used unbounded read: {name}")
        return real_getitem(attrs, name)

    monkeypatch.setattr(h5py.AttributeManager, "__getitem__", guarded)
    with pytest.raises(ValueError, match="text byte ceiling"):
        with FrameViewReader(oversized):
            pass
    with FrameViewReader(ordinary) as reader:
        assert reader._source_base == "/relocated/raw"
        assert reader.labels() == (5,)
    assert probed == []


def test_frame_view_rejects_external_processed_tables(tmp_path):
    from xrd_tools.io.frame_view import FrameViewReader

    foreign = tmp_path / "foreign_tables.h5"
    with h5py.File(foreign, "w") as handle:
        geom = handle.create_group("per_frame_geometry")
        geom.create_dataset("frame_index", data=np.array([5], dtype=np.int64))
        geom.create_dataset("rot1", data=np.array([9.0], dtype=np.float32))
        scan_data = handle.create_group("scan_data")
        scan_data.create_dataset("frame_index", data=np.array([5], dtype=np.int64))
        scan_data.create_dataset("monitor", data=np.array([999.0], dtype=np.float32))
        source = handle.create_group("frames/frame_0005/source")
        source.create_dataset("path", data=np.bytes_("foreign.raw"))
        source.create_dataset("frame_index", data=np.int64(0))

    path = tmp_path / "external_tables.nxs"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        write_integrated_stack(
            entry, frame_indices=[5], results_1d=[_r1d()],
        )
        for name in ("per_frame_geometry", "scan_data", "frames"):
            entry[name] = h5py.ExternalLink(str(foreign), f"/{name}")

    with pytest.raises(ValueError, match="local hard-linked"):
        with FrameViewReader(path):
            pass


def test_frame_view_applies_one_aggregate_budget_across_vlen_columns(
    tmp_path, monkeypatch,
):
    from xrd_tools.io import frame_view as module

    path = tmp_path / "bounded_vlen_aggregate.nxs"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        write_integrated_stack(
            entry, frame_indices=[5], results_1d=[_r1d()],
        )
        scan_data = entry.create_group("scan_data")
        scan_data.create_dataset("frame_index", data=np.array([5], dtype=np.int64))
        for name in ("first", "second"):
            scan_data.create_dataset(
                name,
                data=np.asarray(["123456"], dtype=object),
                dtype=h5py.string_dtype("utf-8"),
            )

    # Each column's retained object/pointer cost is independently admissible;
    # only their aggregate exceeds the shared materialization budget.
    one_column = np.dtype(object).itemsize + max(64, sys.getsizeof("123456"))
    monkeypatch.setattr(module, "_MAX_SCAN_DATA_BYTES", one_column + 1)
    with module.FrameViewReader(path) as reader:
        with pytest.raises(ValueError, match="retained object bytes"):
            reader.read(5)


def test_frame_view_does_not_decode_second_vlen_column_after_budget_spent(
    tmp_path, monkeypatch,
):
    from xrd_tools.io import frame_view as module

    path = tmp_path / "bounded_vlen_remaining_allowance.nxs"
    value = "first"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        write_integrated_stack(
            entry, frame_indices=[5], results_1d=[_r1d()],
        )
        scan_data = entry.create_group("scan_data")
        scan_data.create_dataset("frame_index", data=np.array([5], dtype=np.int64))
        for name, text in (("a_first", value), ("z_second", "must-not-read")):
            scan_data.create_dataset(
                name,
                data=np.asarray([text], dtype=object),
                dtype=h5py.string_dtype("utf-8"),
            )

    first_retained = np.dtype(object).itemsize + max(
        64, sys.getsizeof(value),
    )
    monkeypatch.setattr(module, "_MAX_SCAN_DATA_BYTES", first_retained)
    real_read = module._bounded_utf8_item
    reads = []

    def tracked_read(dataset, row, *, role):
        reads.append(dataset.name)
        return real_read(dataset, row, role=role)

    monkeypatch.setattr(module, "_bounded_utf8_item", tracked_read)
    with module.FrameViewReader(path) as reader:
        with pytest.raises(ValueError, match="retained object bytes"):
            reader.read(5)
    assert reads == ["/entry/scan_data/a_first"]


def test_frame_view_vlen_preflight_charges_empty_python_cells_before_read(
    tmp_path, monkeypatch,
):
    from xrd_tools.io import frame_view as module

    path = tmp_path / "bounded_vlen_empty_cells.nxs"
    rows = 100
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        write_integrated_stack(
            entry, frame_indices=[5], results_1d=[_r1d()],
        )
        scan_data = entry.create_group("scan_data")
        scan_data.create_dataset(
            "frame_index", data=np.arange(rows, dtype=np.int64),
        )
        scan_data.create_dataset(
            "empty",
            data=np.asarray([""] * rows, dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )

    minimum = rows * (np.dtype(object).itemsize + 64)
    monkeypatch.setattr(module, "_MAX_SCAN_DATA_BYTES", minimum - 1)
    monkeypatch.setattr(
        module,
        "_bounded_utf8_item",
        lambda *_args, **_kwargs: pytest.fail(
            "VLEN cell was read before retained-object preflight"
        ),
    )
    with module.FrameViewReader(path) as reader:
        with pytest.raises(ValueError, match="retained object bytes"):
            reader.read(5)


@pytest.mark.parametrize(
    "node", ("entry", "integrated_1d", "nested_mode", "scientific_leaf"),
)
def test_frame_view_rejects_external_scientific_ancestry(tmp_path, node):
    from xrd_tools.io.frame_view import FrameViewReader

    foreign = tmp_path / f"foreign_{node}.h5"
    with h5py.File(foreign, "w") as handle:
        entry = handle.create_group("entry")
        write_integrated_stack(
            entry, frame_indices=[5], results_1d=[_r1d()],
        )
        entry["integrated_1d"].create_group("q_ip")

    path = tmp_path / f"external_{node}.nxs"
    with h5py.File(path, "w") as handle:
        if node == "entry":
            handle["entry"] = h5py.ExternalLink(str(foreign), "/entry")
        else:
            entry = handle.create_group("entry")
            write_integrated_stack(
                entry, frame_indices=[5], results_1d=[_r1d()],
            )
            if node == "integrated_1d":
                del entry["integrated_1d"]
                entry["integrated_1d"] = h5py.ExternalLink(
                    str(foreign), "/entry/integrated_1d",
                )
            elif node == "nested_mode":
                entry["integrated_1d"]["q_ip"] = h5py.ExternalLink(
                    str(foreign), "/entry/integrated_1d/q_ip",
                )
            else:
                del entry["integrated_1d/q"]
                entry["integrated_1d"]["q"] = h5py.ExternalLink(
                    str(foreign), "/entry/integrated_1d/q",
                )

    with pytest.raises(ValueError, match="local hard-linked"):
        with FrameViewReader(path):
            pass


@pytest.mark.parametrize(
    "case", ("oversize", "wrong_dtype", "external_storage", "external_mask"),
)
def test_frame_view_refuses_unbounded_or_foreign_thumbnail_before_getitem(
    tmp_path, monkeypatch, case,
):
    from xrd_tools.io.frame_view import FrameViewReader

    foreign = tmp_path / f"foreign_thumbnail_{case}.h5"
    with h5py.File(foreign, "w") as handle:
        handle.create_dataset("mask", data=np.zeros((2, 2), dtype=bool))

    path = tmp_path / f"bounded_thumbnail_{case}.nxs"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        write_integrated_stack(
            entry, frame_indices=[5], results_1d=[_r1d()],
        )
        frame = entry.create_group("frames/frame_0005")
        if case == "oversize":
            thumbnail = frame.create_dataset(
                "thumbnail", shape=(257, 1), dtype=np.uint8,
            )
        elif case == "wrong_dtype":
            thumbnail = frame.create_dataset(
                "thumbnail", data=np.zeros((2, 2), dtype=np.float32),
            )
        elif case == "external_storage":
            thumbnail = frame.create_dataset(
                "thumbnail",
                shape=(2, 2),
                dtype=np.uint8,
                external=[("thumbnail.raw", 0, 4)],
            )
        else:
            thumbnail = frame.create_dataset(
                "thumbnail", data=np.zeros((2, 2), dtype=np.uint8),
            )
            frame["thumbnail_mask"] = h5py.ExternalLink(
                str(foreign), "/mask",
            )
        thumbnail.attrs["vmin"] = 0.0
        thumbnail.attrs["vmax"] = 1.0
        thumbnail.attrs["dtype"] = thumbnail.dtype.name

    real_getitem = h5py.Dataset.__getitem__
    probed = []

    def guarded(dataset, selection):
        if dataset.name.endswith("/thumbnail"):
            probed.append(selection)
            raise AssertionError("thumbnail was read before qualification")
        return real_getitem(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    with FrameViewReader(path) as reader:
        with pytest.raises(ValueError, match="bounded|local hard-linked"):
            reader.read(5)
    assert probed == []


def test_frame_view_accepts_bounded_uint16_thumbnail_and_direct_mask(tmp_path):
    path = tmp_path / "bounded_uint16_thumbnail.nxs"
    encoded = np.array([[0, 65535], [32768, 16384]], dtype=np.uint16)
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        write_integrated_stack(
            entry, frame_indices=[5], results_1d=[_r1d()],
        )
        frame = entry.create_group("frames/frame_0005")
        thumbnail = frame.create_dataset("thumbnail", data=encoded)
        thumbnail.attrs["vmin"] = 10.0
        thumbnail.attrs["vmax"] = 20.0
        thumbnail.attrs["dtype"] = "uint16"
        frame.create_dataset(
            "thumbnail_mask",
            data=np.array([[False, True], [False, False]], dtype=bool),
        )

    loaded = read_frame_view(path, 5)
    assert loaded.thumbnail is not None
    assert loaded.thumbnail.shape == (2, 2)
    assert loaded.thumbnail[0, 0] == pytest.approx(10.0)
    assert np.isnan(loaded.thumbnail[0, 1])
    assert loaded.thumbnail[1, 0] == pytest.approx(
        10.0 + (32768.0 / 65535.0) * 10.0,
    )


def test_browse_presentation_is_exact_bounded_projection(tmp_path):
    from xrd_tools.core.provenance import read_provenance, write_provenance
    from xrd_tools.io.browse_presentation import read_browse_presentation

    path = tmp_path / "presentation.nxs"
    config = {
        "poni_file": "/calibration/beam.poni",
        "gi": {"enabled": True, "resolved_motor": "th"},
        "mask_file": "/calibration/mask.edf",
        "geometry": {
            "convention": "psic",
            "mapping_json": '{"incident_angle":{"source_motor":"th"}}',
        },
        "unrelated": {"large": ["not traversed"] * 20},
    }
    with h5py.File(path, "w") as handle:
        write_provenance(
            handle, config=config, inputs={"raw_files": ["raw.h5"]},
            program_version="test", host="",
        )
        mono = handle["entry"].create_group("instrument/monochromator")
        mono.create_dataset("wavelength", data=np.float64(1.239841984))

    full = read_provenance(path)["config"]
    presentation, mask = read_browse_presentation(path)
    assert presentation == {
        "poni_file": full["poni_file"],
        "gi": full["gi"],
        "geometry": full["geometry"],
        "wavelength_m": pytest.approx(1.239841984e-10),
    }
    assert mask == full["mask_file"]
    assert "unrelated" not in presentation


def test_browse_presentation_does_not_follow_external_geometry(tmp_path):
    from xrd_tools.core.provenance import write_provenance
    from xrd_tools.io.browse_presentation import read_browse_presentation

    foreign = tmp_path / "foreign.h5"
    with h5py.File(foreign, "w") as handle:
        group = handle.create_group("geometry")
        group.create_dataset(
            "mapping_json", data="{}", dtype=h5py.string_dtype("utf-8"),
        )
    path = tmp_path / "external_geometry.nxs"
    with h5py.File(path, "w") as handle:
        write_provenance(
            handle, config={"poni_file": "beam.poni"},
            program_version="test", host="",
        )
        handle["entry/reduction/config"]["geometry"] = h5py.ExternalLink(
            str(foreign), "/geometry",
        )
    with pytest.raises(ValueError, match="local hard-linked"):
        read_browse_presentation(path)


@pytest.mark.parametrize("node", ("config", "config_leaf", "monochromator"))
def test_browse_presentation_rejects_external_ancestry(tmp_path, node):
    from xrd_tools.core.provenance import write_provenance
    from xrd_tools.io.browse_presentation import read_browse_presentation

    foreign = tmp_path / f"foreign_presentation_{node}.h5"
    with h5py.File(foreign, "w") as handle:
        if node in {"config", "config_leaf"}:
            group = handle.create_group("config")
            group.create_dataset(
                "poni_file",
                data='"foreign.poni"',
                dtype=h5py.string_dtype("utf-8"),
            )
        else:
            group = handle.create_group("monochromator")
            group.create_dataset("wavelength", data=np.float64(1.0))

    path = tmp_path / f"external_presentation_{node}.nxs"
    with h5py.File(path, "w") as handle:
        write_provenance(
            handle, config={"poni_file": "beam.poni"},
            program_version="test", host="",
        )
        entry = handle["entry"]
        if node == "config":
            del entry["reduction/config"]
            entry["reduction"]["config"] = h5py.ExternalLink(
                str(foreign), "/config",
            )
        elif node == "config_leaf":
            del entry["reduction/config/poni_file"]
            entry["reduction/config"]["poni_file"] = h5py.ExternalLink(
                str(foreign), "/config/poni_file",
            )
        else:
            instrument = entry.create_group("instrument")
            instrument["monochromator"] = h5py.ExternalLink(
                str(foreign), "/monochromator",
            )

    with pytest.raises(ValueError, match="local hard-linked"):
        read_browse_presentation(path)


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
