"""Headless scan grouping + composite source + directory discovery."""
import numpy as np
import pytest

from xrd_tools.core.scan import ScanFrame, SourceKind
from xrd_tools.sources import (CompositeFrameSource, MemoryFrameSource,
                               discover_scans, flatten_scan_groups,
                               parse_scan_groups)


# ── parse_scan_groups ─────────────────────────────────────────────────────


def test_parse_scan_groups_combines_ranges():
    assert parse_scan_groups("1-3, 5, 7-9") == [[1, 2, 3], [5], [7, 8, 9]]
    assert parse_scan_groups(" 4 ") == [[4]]
    assert parse_scan_groups("3-1") == [[1, 2, 3]]            # reversed normalised
    assert parse_scan_groups("1, , 2") == [[1], [2]]          # empty token skipped
    assert flatten_scan_groups([[1, 2, 3], [2, 5]]) == [1, 2, 3, 5]   # de-duped
    with pytest.raises(ValueError):
        parse_scan_groups("1-x")


# ── CompositeFrameSource ──────────────────────────────────────────────────


def _mem(values, *, meta=None):
    frames = [ScanFrame(index=i, image=np.full((3, 3), v, dtype=float),
                        metadata=(meta[i] if meta else {}))
              for i, v in enumerate(values)]
    return MemoryFrameSource(frames)


def test_composite_concatenates_frames_and_dispatches():
    a = _mem([1, 2], meta=[{"t": 10}, {"t": 11}])
    b = _mem([3, 4, 5], meta=[{"t": 20}, {"t": 21}, {"t": 22}])
    comp = CompositeFrameSource([a, b])
    assert comp.frame_indices == [0, 1, 2, 3, 4]              # 2 + 3 re-indexed
    # load_frame dispatches to the owning member
    np.testing.assert_allclose(comp.load_frame(0), 1.0)
    np.testing.assert_allclose(comp.load_frame(2), 3.0)      # b's first
    np.testing.assert_allclose(comp.load_frame(4), 5.0)
    # metadata dispatches
    assert dict(comp.metadata_for(1))["t"] == 11
    assert dict(comp.metadata_for(3))["t"] == 21


def test_composite_motors_concatenate_with_nan_pad():
    class _Motors(MemoryFrameSource):
        def __init__(self, imgs, motors):
            super().__init__(imgs)
            self._mt = motors

        @property
        def motors(self):
            return self._mt

    a = _Motors([np.zeros((2, 2))] * 2, {"th": np.array([0.0, 1.0])})
    b = _Motors([np.zeros((2, 2))] * 2,
                {"th": np.array([2.0, 3.0]), "chi": np.array([9.0, 9.0])})
    comp = CompositeFrameSource([a, b])
    np.testing.assert_allclose(comp.motors["th"], [0, 1, 2, 3])
    # 'chi' only in b -> NaN block for a's frames
    chi = comp.motors["chi"]
    assert np.isnan(chi[0]) and np.isnan(chi[1])
    np.testing.assert_allclose(chi[2:], [9, 9])


def test_composite_motors_clipped_to_frame_count():
    """A member whose motor array is LONGER than its frame count (a partial scan
    with fewer images than metadata points) is clipped, so each per-key column
    stays frame-aligned (len == len(frame_indices))."""
    class _Motors(MemoryFrameSource):
        def __init__(self, imgs, motors):
            super().__init__(imgs)
            self._mt = motors

        @property
        def motors(self):
            return self._mt

    a = _Motors([np.zeros((2, 2))] * 2,          # 2 frames
                {"th": np.array([0.0, 1.0, 99.0])})   # 3-long motor array
    comp = CompositeFrameSource([a])
    assert len(comp.motors["th"]) == len(comp.frame_indices) == 2
    np.testing.assert_allclose(comp.motors["th"], [0.0, 1.0])   # 99 clipped off


def test_composite_motors_cached_and_returned_as_copies():
    class _Motors(MemoryFrameSource):
        def __init__(self, imgs, motors):
            super().__init__(imgs)
            self._mt = motors
            self.calls = 0

        @property
        def motors(self):
            self.calls += 1
            return self._mt

    a = _Motors([np.zeros((2, 2))], {"th": np.array([1.0])})
    b = _Motors([np.zeros((2, 2))], {"th": np.array([2.0])})
    comp = CompositeFrameSource([a, b])

    first = comp.motors
    first["th"][0] = 99.0
    second = comp.motors

    np.testing.assert_allclose(second["th"], [1.0, 2.0])
    assert a.calls == 1 and b.calls == 1


def test_composite_dispatches_sparse_labels():
    """A member with non-0..n-1 frame labels (e.g. 5, 7) still dispatches
    load_frame/metadata_for to the right member frame."""
    a = MemoryFrameSource([
        ScanFrame(index=5, image=np.full((2, 2), 50.0), metadata={"t": 5}),
        ScanFrame(index=7, image=np.full((2, 2), 70.0), metadata={"t": 7})])
    assert a.frame_indices == [5, 7]
    comp = CompositeFrameSource([a, _mem([1])])
    assert comp.frame_indices == [0, 1, 2]              # re-indexed 0..2
    np.testing.assert_allclose(comp.load_frame(0), 50.0)   # → member label 5
    assert dict(comp.metadata_for(1))["t"] == 7            # → member label 7


def test_composite_accepts_specs(tmp_path):
    """concat_sources([spec, …]) opens the specs internally (the grouping path)."""
    pytest.importorskip("silx")
    from xrd_tools.core.scan import SourceSpec
    from xrd_tools.sources import concat_sources

    spec_file = tmp_path / "myscan"
    spec_file.write_text(
        "#F myscan\n#E 1\n#O0 th\n\n"
        "#S 5 ascan th 0 1 1 1\n#P0 0\n#N 2\n#L th  i0\n0 100\n1 110\n")
    spec = SourceSpec(spec_file, SourceKind.SPEC, options={"scan": "5"})
    comp = concat_sources([spec, _mem([1, 2, 3])])
    assert comp.frame_indices == [0, 1, 2, 3, 4]       # 2 (spec) + 3 (mem)
    assert type(comp.members[0]).__name__ == "SpecSource"


def test_composite_raw_capability_requires_all_members():
    raw = _mem([1])                                  # MemoryFrameSource loads images
    assert CompositeFrameSource([raw, raw]).capabilities.has_raw_references is False
    # MemoryFrameSource advertises has_raw_references=False, so the composite does too;
    # the point: the composite ANDs the members (one no-raw member disables the group).


# ── discover_scans ────────────────────────────────────────────────────────


def test_discover_scans_nexus_and_images(tmp_path):
    (tmp_path / "a.nxs").write_bytes(b"")
    (tmp_path / "b.h5").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x")
    nx = discover_scans(tmp_path, "nexus_stack")
    assert sorted(str(s.uri.name) for s in nx) == ["a.nxs", "b.h5"]
    assert all(s.kind is SourceKind.NEXUS_STACK for s in nx)

    (tmp_path / "img_0001.tif").write_bytes(b"")
    img = discover_scans(tmp_path, "tiff_series")
    assert len(img) == 1 and img[0].kind is SourceKind.TIFF_SERIES

    assert discover_scans(tmp_path / "nope", "nexus_stack") == []


def test_discover_scans_uses_natural_file_order(tmp_path):
    for name in ("scan_10.nxs", "scan_2.nxs", "scan_1.nxs"):
        (tmp_path / name).write_bytes(b"")

    specs = discover_scans(tmp_path, "nexus_stack")

    assert [s.uri.name for s in specs] == [
        "scan_1.nxs",
        "scan_2.nxs",
        "scan_10.nxs",
    ]


def test_discover_scans_eiger_filters_master(tmp_path):
    """Eiger discovery returns only the _master file, not the sibling data files."""
    (tmp_path / "scan_master.h5").write_bytes(b"")
    (tmp_path / "scan_data_000001.h5").write_bytes(b"")
    (tmp_path / "scan_data_000002.h5").write_bytes(b"")
    specs = discover_scans(tmp_path, "eiger_master")
    assert [s.uri.name for s in specs] == ["scan_master.h5"]
    assert specs[0].kind is SourceKind.EIGER_MASTER


def test_discover_scans_spec_per_scan(tmp_path):
    pytest.importorskip("silx")
    spec = tmp_path / "myscan"
    spec.write_text(
        "#F myscan\n#E 1\n#O0 th\n\n"
        "#S 5 ascan th 0 1 1 1\n#P0 0\n#N 2\n#L th  i0\n0 100\n1 110\n\n"
        "#S 6 ascan th 0 1 1 1\n#P0 0\n#N 2\n#L th  i0\n0 200\n1 210\n")
    specs = discover_scans(tmp_path, "spec")
    assert [s.options["scan"] for s in specs] == ["5.1", "6.1"]
    assert all(s.kind is SourceKind.SPEC for s in specs)
