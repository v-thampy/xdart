"""Independent product-boundary discriminators for the expanded acquisition readers."""
import h5py
import numpy as np
import pytest

from xrd_tools.io.image import count_frames, read_image, read_image_stack
from xrd_tools.sources.execution_graph import qualify_source_execution_graph, open_source_execution_graph
from xrd_tools.sources.spec import SourceSpec, SourceKind
from xrd_tools.sources.cursor import ContainerCursor
from xrd_tools.sources.metadata_provider import MetadataSourceClosedError


def test_average_first_row_does_not_read_a_complete_scan_vector(tmp_path, monkeypatch):
    path = tmp_path / "acquisition.nxs"
    frames = 4096
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("instrument/detector/data", shape=(frames, 8, 8), dtype="u2", chunks=(1, 8, 8))
        eta = entry.create_dataset("instrument/positioners/eta/value", data=np.arange(frames, dtype=float), chunks=(32,))
        eta.attrs["units"] = "deg"
    graph = qualify_source_execution_graph(SourceSpec(path, SourceKind.NEXUS_STACK), selected_motor="eta", reader_binding="average_closed_v1")
    reads = []
    original = h5py.Dataset.__getitem__
    def observing_read(dataset, selection):
        if dataset.name == "/entry/instrument/positioners/eta/value":
            reads.append(selection)
        return original(dataset, selection)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", observing_read)
    with open_source_execution_graph(graph) as window:
        assert window.complete_metadata_for(1)["eta"] == 1.0
    assert reads and all(selection != () for selection in reads), f"first row read the full {frames}-element motor array: {reads!r}"


def test_an_external_counter_does_not_replace_local_detector_images(tmp_path):
    path = tmp_path / "scan.h5"
    member = tmp_path / "counter.h5"
    pixels = np.arange(2 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8)
    with h5py.File(member, "w") as handle:
        handle.create_dataset("i0", data=[100., 110.])
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.attrs["signal"] = "data"
        data.create_dataset("data", data=pixels)
        data["i0"] = h5py.ExternalLink(member.name, "/i0")
    assert count_frames(path) == 2
    np.testing.assert_array_equal(read_image(path), pixels[0])


@pytest.mark.parametrize("suffix", [".h5", ".hdf5", ".nxs", ".nexus", ".cxi"])
def test_split_detector_images_do_not_depend_on_the_nexus_suffix(tmp_path, suffix):
    from xrd_tools.io.image import read_detector_image_layout
    from xrd_tools.io.image_source import classify_image_source
    path = tmp_path / f"scan{suffix}"
    pixels = np.arange(4 * 8 * 8, dtype=np.uint16).reshape(4, 8, 8)
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("instrument/detector/detectorSpecific/eiger_fw_version", data=np.bytes_("test"))
        data = entry.create_group("data")
        for index in range(2):
            member = tmp_path / f"pixels_{index}.h5"
            with h5py.File(member, "w") as part:
                part.create_dataset("images", data=pixels[index*2:(index+1)*2])
            data[f"data_{index+1:06d}"] = h5py.ExternalLink(member.name, "/images")
    assert count_frames(path) == 4
    np.testing.assert_array_equal(read_image_stack(path), pixels)
    np.testing.assert_array_equal(read_image(path, frame=3), pixels[3])
    assert read_detector_image_layout(path).frame_count == 4
    assert classify_image_source(path).n_frames == 4


def test_streamed_generic_metadata_seals_on_close_but_materialized_table_survives(tmp_path):
    path = tmp_path / "rows.nxs"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("instrument/detector/data", data=np.zeros((3, 8, 8), np.uint16))
        eta = entry.create_dataset("sample/positioners/eta/value", data=[0., np.pi / 2, np.pi])
        eta.attrs["units"] = "rad"
        entry.create_dataset("instrument/positioners/chi/value", data=12.5)
        entry.create_dataset("data/i0", data=[100., 110., 120.])
        energy = entry.create_dataset("instrument/monochromator/energy", data=[16., 16.001, 16.002])
        energy.attrs["units"] = "keV"
    with ContainerCursor(path) as cursor:
        streamed = cursor.metadata_provider()
        assert dict(streamed.complete_metadata_for(1)) == pytest.approx(
            {"eta": 90., "chi": 12.5, "i0": 110., "energy": 16001.})
        assert streamed.complete_metadata_for(-1) == {}
        assert streamed.complete_metadata_for(3) == {}
    with pytest.raises(MetadataSourceClosedError):
        streamed.complete_metadata_for(1)
    with ContainerCursor(path) as cursor:
        materialized = cursor.metadata_provider()
        first = dict(materialized.complete_metadata_for(1))
        materialized.scan_table()
    assert materialized.complete_metadata_for(1) == first


def test_linked_image_target_landing_is_seen_without_master_stat_change(tmp_path):
    from xrd_tools.io.image import eiger_content
    path, member = tmp_path / "scan.h5", tmp_path / "images.h5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("entry/data")
        data["image"] = h5py.ExternalLink(member.name, "/images")
    state = path.stat()
    eiger_content(path)
    pixels = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)
    with h5py.File(member, "w") as handle:
        handle.create_dataset("images", data=pixels)
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (state.st_size, state.st_mtime_ns)
    assert eiger_content(path)[1]
    np.testing.assert_array_equal(read_image_stack(path), pixels)
