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


def _positioner_entry(path, frames, motors, offset=0.):
    """A raw stack whose entry holds ``motors`` per-frame positioners, m<i> = frame + i + offset (degrees)."""
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("instrument/detector/data", shape=(frames, 8, 8), dtype="u2", chunks=(1, 8, 8))
        for index in range(motors):
            value = entry.create_dataset(f"instrument/positioners/m{index}/value",
                                         data=np.arange(frames, dtype=float) + index + offset, chunks=(32,))
            value.attrs["units"] = "deg"
    return path


def _datasets_in(value):
    """Every h5py dataset reachable from ``value`` through containers."""
    if isinstance(value, h5py.Dataset):
        return [value]
    if isinstance(value, dict):
        return [found for item in value.items() for found in _datasets_in(item)]
    if isinstance(value, (tuple, list, set, frozenset)):
        return [found for item in value for found in _datasets_in(item)]
    return []


def test_streamed_rows_plan_and_admit_once_per_window_within_b4s_bounds(tmp_path, monkeypatch):
    """NEXT-129-SMALL-FIXES item 8 (pre-existing, measured; CAL-HELD-GENERIC-ROW-COST): every streamed row re-planned the
    entry's acquisition items and re-ran Average's admission of each dataset (about 200 us per item-row, 4 ms per row
    with twenty positioners). A window now plans once and admits each dataset once. B4's bounds hold: every row still
    reads its own values from the file (no row cache), the provider holds no dataset, and another file's window plans
    and serves its own items."""
    from xrd_tools.io import bluesky_nexus, nexus
    frames, motors = 64, 20
    plans, admissions, reads = [], [], []
    plan, admit = nexus.acquisition_items, bluesky_nexus._average_fixed_numeric_dataset
    monkeypatch.setattr(nexus, "acquisition_items", lambda *args, **kwargs: plans.append(1) or plan(*args, **kwargs))
    monkeypatch.setattr(bluesky_nexus, "_average_fixed_numeric_dataset",
                        lambda value, **kwargs: admissions.append(value.name) or admit(value, **kwargs))
    original = h5py.Dataset.__getitem__
    def observing_read(dataset, selection, *args, **kwargs):
        if dataset.name == "/entry/instrument/positioners/m3/value":
            reads.append(selection)
        return original(dataset, selection, *args, **kwargs)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", observing_read)
    for name, count, offset in (("a.nxs", motors, 0.), ("b.nxs", 4, 100.)):
        path = _positioner_entry(tmp_path / name, frames, count, offset)
        graph = qualify_source_execution_graph(SourceSpec(path, SourceKind.NEXUS_STACK), selected_motor="m0",
                                               reader_binding="average_closed_v1")
        with open_source_execution_graph(graph) as window:
            first = window.complete_metadata_for(0)  # the window's provider plans and admits here
            plans.clear(); admissions.clear(); reads.clear()
            rows = [first] + [window.complete_metadata_for(index) for index in range(1, frames)]
            held = _datasets_in(vars(window._cursor.metadata_provider()))
        assert rows == [{f"m{motor}": float(index + motor + offset) for motor in range(count)}
                        for index in range(frames)]
        assert plans == [], f"{name}: {len(plans)} streamed rows planned the entry's items again"
        assert admissions == [], f"{name}: {len(admissions)} datasets were admitted again"
        assert reads == list(range(1, frames)), f"{name}: rows did not each read their own values: {reads!r}"
        assert held == [], f"{name}: the provider holds datasets {held!r}"
