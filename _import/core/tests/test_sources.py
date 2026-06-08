from __future__ import annotations

import h5py
import numpy as np
import pytest


def test_memory_source_round_trips_to_scan():
    from ssrl_xrd_tools.core.scan import ScanFrame
    from ssrl_xrd_tools.sources import MemoryFrameSource

    source = MemoryFrameSource([
        ScanFrame(3, image=np.full((2, 2), 3), metadata={"tag": "c"}),
        ScanFrame(1, image=np.ones((2, 2)), metadata={"tag": "a"}),
    ])

    assert source.frame_indices == [1, 3]
    assert np.array_equal(source.load_frame(3), np.full((2, 2), 3))
    scan = source.to_scan(name="from_source")
    assert scan.name == "from_source"
    assert list(scan.to_scan_data()["tag"]) == ["a", "c"]


def test_live_source_append_and_duplicate_guard():
    from ssrl_xrd_tools.sources import LiveFrameSource

    source = LiveFrameSource()
    source.append(np.ones((2, 2)), index=10, metadata={"i0": 5})
    assert source.frame_indices == [10]
    assert source.metadata_for(10)["i0"] == 5
    with pytest.raises(ValueError, match="duplicate live frame"):
        source.append(np.zeros((2, 2)), index=10)


def test_nexus_stack_source_loads_chunks(tmp_path):
    from ssrl_xrd_tools.sources import NexusStackSource, open_source

    path = tmp_path / "raw_stack.nxs"
    data = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        inst = entry.create_group("instrument")
        det = inst.create_group("detector")
        det.create_dataset("data", data=data)

    source = NexusStackSource(path)
    assert source.frame_indices == [0, 1, 2]
    assert np.array_equal(source.load_frame(2), data[2])
    chunks = list(source.iter_chunks(2))
    assert [labels for _, labels in chunks] == [[0, 1], [2]]
    assert np.array_equal(chunks[0][0], data[:2])

    guessed = open_source(path)
    assert isinstance(guessed, NexusStackSource)


def test_image_file_source_uses_existing_reader(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    from ssrl_xrd_tools.sources import ImageFileSource, open_source

    path = tmp_path / "frame.tif"
    image = np.arange(12, dtype=np.uint16).reshape(3, 4)
    tifffile.imwrite(path, image)

    source = ImageFileSource(path)
    assert source.frame_indices == [0]
    assert np.array_equal(source.load_frame(0), image)
    assert isinstance(open_source(path), ImageFileSource)


def test_tiff_series_from_directory_uses_natural_order_and_pattern(tmp_path):
    from ssrl_xrd_tools.sources import TiffSeriesSource

    for name in (
        "scan_1.tif",
        "scan_10.tif",
        "scan_2.tif",
        "scan_9.tif",
        "other_3.tif",
        "scan_4.edf",
        "scan_11.txt",
    ):
        (tmp_path / name).touch()

    source = TiffSeriesSource.from_directory(tmp_path, pattern="scan_*.tif")

    assert [path.name for path in source.files] == [
        "scan_1.tif",
        "scan_2.tif",
        "scan_9.tif",
        "scan_10.tif",
    ]
    assert source.frame_indices == [1, 2, 3, 4]


def test_processed_nexus_source_reads_frame_views(tmp_path):
    from ssrl_xrd_tools.sources import ProcessedNexusSource, open_source

    path = tmp_path / "processed.nxs"
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        g1 = entry.create_group("integrated_1d")
        g1.create_dataset("frame_index", data=np.array([7], dtype=np.int64))
        q = g1.create_dataset("q", data=np.linspace(0.1, 1.0, 5))
        q.attrs["units"] = "q_A^-1"
        g1.create_dataset("intensity", data=np.arange(5, dtype=np.float32)[None, :])
        sd = entry.create_group("scan_data")
        sd.create_dataset("frame_index", data=np.array([7], dtype=np.int64))
        sd.create_dataset("i0", data=np.array([11.0], dtype=np.float32))

    source = ProcessedNexusSource(path)
    assert source.frame_indices == [7]
    view = source.read_view(7)
    assert view.has_1d
    assert np.allclose(view.intensity_1d, np.arange(5))
    assert view.metadata_numeric["i0"] == 11.0
    assert isinstance(open_source(path), ProcessedNexusSource)
