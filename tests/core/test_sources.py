from __future__ import annotations

import h5py
import numpy as np
import pytest


def test_memory_source_round_trips_to_scan():
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.sources import MemoryFrameSource

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
    from xrd_tools.sources import LiveFrameSource

    source = LiveFrameSource()
    source.append(np.ones((2, 2)), index=10, metadata={"i0": 5})
    assert source.frame_indices == [10]
    assert source.metadata_for(10)["i0"] == 5
    with pytest.raises(ValueError, match="duplicate live frame"):
        source.append(np.zeros((2, 2)), index=10)


def test_nexus_stack_source_loads_chunks(tmp_path):
    from xrd_tools.sources import NexusStackSource, open_source

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
    from xrd_tools.sources import ImageFileSource, open_source

    path = tmp_path / "frame.tif"
    image = np.arange(12, dtype=np.uint16).reshape(3, 4)
    tifffile.imwrite(path, image)

    source = ImageFileSource(path)
    assert source.frame_indices == [0]
    assert np.array_equal(source.load_frame(0), image)
    assert isinstance(open_source(path), ImageFileSource)


def test_image_file_source_reads_headerless_raw_without_frame_count_warning(
        tmp_path, caplog):
    from xrd_tools.sources import ImageFileSource

    path = tmp_path / "scan_0000.raw"
    image = np.arange(3 * 4, dtype=np.uint16).reshape(3, 4)
    image.tofile(path)

    source = ImageFileSource(
        path,
        detector_shape=(3, 4),
        raw_dtype="uint16",
        raw_header_skip=0,
    )

    assert source.frame_indices == [0]
    np.testing.assert_array_equal(source.load_frame(0), image)
    assert "Could not determine frame count" not in caplog.text


def test_common_headerless_raw_shape_is_inferred_without_gui_defaults(tmp_path):
    from xrd_tools.io.image import infer_raw_detector_shape, read_image
    from xrd_tools.sources import ImageFileSource, TiffSeriesSource

    shape = (195, 487)
    images = []
    for index in (1, 2):
        image = np.full(shape, index, dtype=np.int32)
        image.tofile(tmp_path / f"scan_{index:04d}.raw")
        images.append(image)

    first = tmp_path / "scan_0001.raw"
    assert infer_raw_detector_shape(first) == shape
    np.testing.assert_array_equal(read_image(first), images[0])
    np.testing.assert_array_equal(ImageFileSource(first).load_frame(0), images[0])

    series = TiffSeriesSource.from_directory(
        tmp_path, pattern="scan_*.raw", metadata_format=None)
    np.testing.assert_array_equal(series.load_frame(2), images[1])


def test_unknown_headerless_raw_shape_is_not_guessed(tmp_path):
    from xrd_tools.io.image import infer_raw_detector_shape

    path = tmp_path / "unknown.raw"
    np.zeros((17, 19), dtype=np.int32).tofile(path)

    assert infer_raw_detector_shape(path) is None


def test_tiff_series_from_directory_uses_natural_order_and_pattern(tmp_path):
    from xrd_tools.sources import TiffSeriesSource

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


def test_raw_series_from_directory_threads_binary_read_parameters(tmp_path):
    from xrd_tools.sources import TiffSeriesSource

    for index in (1, 10, 2):
        np.full((3, 4), index, dtype=np.uint16).tofile(
            tmp_path / f"scan_{index:04d}.raw")

    source = TiffSeriesSource.from_directory(
        tmp_path,
        pattern="scan_*.raw",
        metadata_format=None,
        detector_shape=(3, 4),
        raw_dtype="uint16",
        raw_header_skip=0,
    )

    assert [path.name for path in source.files] == [
        "scan_0001.raw",
        "scan_0002.raw",
        "scan_0010.raw",
    ]
    assert source.frame_indices == [1, 2, 3]
    np.testing.assert_array_equal(source.load_frame(2), np.full((3, 4), 2))
    np.testing.assert_array_equal(
        source.frame_for(3).load_image(), np.full((3, 4), 10))


def test_processed_nexus_source_reads_frame_views(tmp_path):
    from xrd_tools.sources import ProcessedNexusSource, open_source

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
