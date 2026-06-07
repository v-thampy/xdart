from __future__ import annotations

import h5py
import numpy as np

from ssrl_xrd_tools.core import IntegrationResult1D, IntegrationResult2D, TwoDKind
from ssrl_xrd_tools.io import (
    inspect_nexus,
    preview_nexus_dataset,
    read_nexus_dataset,
)
from ssrl_xrd_tools.io.nexus import write_integrated_stack


def _r1d() -> IntegrationResult1D:
    return IntegrationResult1D(
        radial=np.linspace(0.5, 2.5, 5),
        intensity=np.linspace(10.0, 50.0, 5),
        sigma=np.ones(5),
        unit="q_A^-1",
    )


def _gi_2d() -> IntegrationResult2D:
    return IntegrationResult2D(
        radial=np.linspace(-1.0, 1.0, 4),
        azimuthal=np.linspace(0.0, 3.0, 3),
        intensity=np.arange(12, dtype=float).reshape(4, 3),
        unit="qip_A^-1",
        azimuthal_unit="qoop_A^-1",
    )


def test_inspect_nexus_summarizes_xdart_schema_without_loading_arrays(tmp_path):
    path = tmp_path / "inspectable.nxs"
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        write_integrated_stack(
            entry,
            frame_indices=[2, 4],
            results_1d=[_r1d(), _r1d()],
            results_2d=[_gi_2d(), _gi_2d()],
        )
        scan_data = entry.create_group("scan_data")
        scan_data.create_dataset("frame_index", data=np.array([2, 4]))
        scan_data.create_dataset("th", data=np.array([0.1, 0.2]))
        geom = entry.create_group("per_frame_geometry")
        geom.create_dataset("frame_index", data=np.array([2, 4]))
        geom.create_dataset("incident_angle", data=np.array([0.1, 0.2]))
        frame = entry.create_group("frames/frame_0002")
        frame.create_dataset("thumbnail", data=np.ones((2, 2)))
        source = frame.create_group("source")
        source.create_dataset("path", data=np.bytes_("raw.h5"))
        det = entry.create_group("instrument/detector")
        det.create_dataset("data", data=np.zeros((2, 8, 8), dtype=np.uint16))

    summary = inspect_nexus(path, max_depth=3)

    assert summary.entries == ("entry",)
    assert summary.tree.kind == "group"
    assert summary.xdart is not None
    assert summary.xdart.is_processed
    assert summary.xdart.frame_labels == (2, 4)
    assert summary.xdart.raw_image_dataset == "/entry/instrument/detector/data"
    assert summary.xdart.raw_image_shape == (2, 8, 8)
    assert summary.xdart.raw_image_dtype == "uint16"
    assert summary.xdart.scan_data_columns == ("th",)
    assert summary.xdart.geometry_columns == ("incident_angle",)
    assert summary.xdart.thumbnail_count == 1
    assert summary.xdart.source_count == 1

    one_d = summary.xdart.integrated_1d
    assert one_d is not None
    assert one_d.frame_count == 2
    assert one_d.intensity_shape == (2, 5)
    assert one_d.axes[0].units == "q_A^-1"

    two_d = summary.xdart.integrated_2d
    assert two_d is not None
    assert two_d.two_d_kind is TwoDKind.QIP_QOOP
    assert two_d.intensity_shape == (2, 3, 4)
    assert [axis.units for axis in two_d.axes] == ["qip_A^-1", "qoop_A^-1"]


def test_preview_nexus_dataset_returns_bounded_head_slice(tmp_path):
    path = tmp_path / "preview.nxs"
    with h5py.File(path, "w") as h5:
        ds = h5.create_dataset("entry/data", data=np.arange(100).reshape(10, 10))
        ds.attrs["units"] = "counts"

    preview = preview_nexus_dataset(path, "/entry/data", max_items=9)

    assert preview.path == "/entry/data"
    assert preview.shape == (10, 10)
    assert preview.dtype.startswith("int")
    assert preview.selection == "[:3, :3]"
    assert preview.truncated
    assert preview.attrs["units"] == "counts"
    assert preview.data == [[0, 1, 2], [10, 11, 12], [20, 21, 22]]


def test_read_nexus_dataset_can_read_full_or_selected_data(tmp_path):
    path = tmp_path / "read_dataset.nxs"
    data = np.arange(24).reshape(2, 3, 4)
    with h5py.File(path, "w") as h5:
        ds = h5.create_dataset("entry/data", data=data)
        ds.attrs["units"] = "counts"

    full = read_nexus_dataset(path, "/entry/data")
    assert full.shape == (2, 3, 4)
    assert full.selection == "[:, :, :]"
    assert full.attrs["units"] == "counts"
    np.testing.assert_array_equal(full.data, data)

    selected = read_nexus_dataset(path, "/entry/data", selection=np.s_[1, :, ::2])
    assert selected.selection == "[1, :, ::2]"
    np.testing.assert_array_equal(selected.data, data[1, :, ::2])


def test_inspect_nexus_does_not_treat_native_frames_group_as_processed(tmp_path):
    path = tmp_path / "raw_eiger_like.h5"
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_group("frames")
        data = entry.create_group("data")
        data.create_dataset("data", data=np.zeros((3, 4, 5), dtype=np.uint16))

    summary = inspect_nexus(path)

    assert summary.xdart is not None
    assert not summary.xdart.is_processed
    assert summary.xdart.frame_labels == ()
    assert summary.xdart.raw_image_dataset == "/entry/data/data"
    assert summary.xdart.raw_image_shape == (3, 4, 5)
    assert summary.xdart.raw_image_dtype == "uint16"


def test_inspect_nexus_raw_data_wins_over_reduction_only_marker(tmp_path):
    path = tmp_path / "raw_with_reduction_marker.h5"
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_group("reduction")
        entry.create_group("frames/frame_0000")
        data = entry.create_group("data")
        data.create_dataset("data", data=np.zeros((1, 4, 5), dtype=np.uint16))

    summary = inspect_nexus(path)

    assert summary.xdart is not None
    assert not summary.xdart.is_processed
    assert summary.xdart.frame_labels == ()
    assert summary.xdart.raw_image_dataset == "/entry/data/data"
    assert summary.xdart.raw_image_shape == (1, 4, 5)


def test_inspect_nexus_reduction_only_is_not_processed(tmp_path):
    path = tmp_path / "reduction_only_partial.nxs"
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_group("reduction")

    summary = inspect_nexus(path)

    assert summary.xdart is not None
    assert not summary.xdart.is_processed
    assert summary.xdart.frame_labels == ()
    assert summary.xdart.raw_image_dataset is None
    assert summary.xdart.integrated_1d is None
    assert summary.xdart.integrated_2d is None


def test_raw_dataset_search_prunes_frames_subtree(tmp_path):
    path = tmp_path / "fallback_raw_search.h5"
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("frames/frame_0000/native_blob", data=np.zeros((20, 20)))
        entry.create_dataset("other/raw", data=np.zeros((2, 3)))

    summary = inspect_nexus(path)

    assert summary.xdart is not None
    assert summary.xdart.raw_image_dataset == "/entry/other/raw"
