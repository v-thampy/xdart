"""Metadata-only + image-backed SPEC FrameSource: content-based (extensionless)
classification, ALL per-point + constant motors recorded, scan selection, and
optional raw-image loading via the io.image layer."""
import numpy as np
import pytest

pytest.importorskip("silx")

from xrd_tools.io.spec import is_spec_file
from xrd_tools.sources import (SourceKind, SpecSource, guess_source_kind,
                               open_source)

_SPEC = """#F myscan
#E 1
#D today
#O0 th  chi  phi

#S 5 ascan th 0 2 2 1
#D today
#P0 0 5 10
#N 4
#L th  Epoch  i0  det
0 1 100 10
1 2 110 20
2 3 120 30

#S 6 ascan chi 0 1 1 1
#D today
#P0 7 0 10
#N 2
#L chi  i0
0 300
1 310
"""


def _spec(tmp_path):
    p = tmp_path / "myscan"          # extensionless — the SSRL convention
    p.write_text(_SPEC)
    return p


def test_spec_detected_by_content_not_extension(tmp_path):
    p = _spec(tmp_path)
    assert p.suffix == ""                        # no extension
    assert is_spec_file(p) is True
    assert guess_source_kind(p) is SourceKind.SPEC
    # a non-SPEC text file is not misclassified
    other = tmp_path / "notes"
    other.write_text("hello world\n")
    assert is_spec_file(other) is False


def test_spec_records_all_motors_and_columns(tmp_path):
    src = open_source(_spec(tmp_path), scan=5)
    assert isinstance(src, SpecSource)
    assert src.scan_key == "5.1" and src.frame_indices == [0, 1, 2]
    # per-point #L columns (whole arrays)
    np.testing.assert_allclose(src.motors["th"], [0, 1, 2])
    np.testing.assert_allclose(src.motors["i0"], [100, 110, 120])
    # metadata_for merges per-point (winning) with ALL #O/#P motors — incl. the
    # NON-scanned chi/phi (the GI-incidence case the metadata work targets)
    md = dict(src.metadata_for(1))
    assert md["th"] == 1.0 and md["i0"] == 110.0          # per-point
    assert md["chi"] == 5.0 and md["phi"] == 10.0          # constant, non-scanned


def test_spec_metadata_only_disables_raw(tmp_path):
    src = SpecSource(_spec(tmp_path), scan=5)
    assert src.capabilities.has_metadata is True
    assert src.capabilities.has_raw_references is False
    with pytest.raises(NotImplementedError):
        src.load_frame(0)


def test_spec_scan_selection(tmp_path):
    p = _spec(tmp_path)
    assert SpecSource(p).scan_key == "5.1"               # default = first scan
    second = SpecSource(p, scan=6)
    assert second.scan_key == "6.1" and second.frame_indices == [0, 1]
    np.testing.assert_allclose(second.motors["chi"], [0, 1])


def test_spec_with_images_enables_raw(tmp_path):
    """An image directory + the scan's filename stem makes the frames loadable
    raw images (ROI/stitch/RSM) — format-agnostic via io.image.read_image."""
    p = _spec(tmp_path)
    for i in range(3):                               # one raw file per frame
        (np.full((6, 6), i + 1, dtype="int32")).tofile(
            tmp_path / f"myscan_scan5_{i:04d}.raw")
    src = SpecSource(p, scan=5, image_dir=tmp_path,
                     read_image_kwargs={"detector_shape": (6, 6),
                                        "raw_dtype": "int32"})
    assert src.capabilities.has_raw_references is True
    assert src.frame_indices == [0, 1, 2]
    np.testing.assert_allclose(src.load_frame(0), 1.0)
    np.testing.assert_allclose(src.load_frame(2), 3.0)
    # scan 5's stem must NOT pick up scan 50's images (trailing-_ anchor)
    (np.zeros((6, 6), dtype="int32")).tofile(tmp_path / "myscan_scan50_0000.raw")
    src2 = SpecSource(p, scan=5, image_dir=tmp_path,
                      read_image_kwargs={"detector_shape": (6, 6),
                                         "raw_dtype": "int32"})
    assert src2.frame_indices == [0, 1, 2]           # still only scan 5's three


def test_spec_frame_for_carries_raw_source_pointer(tmp_path):
    """frame_for must attach the raw-image source pointer (the stitch/RSM raw-popup
    enabler) — else a SPEC-sourced stitch persists empty contributing-frame
    records and the popup silently vanishes for the commonest stitch source."""
    p = _spec(tmp_path)
    for i in range(3):
        (np.full((6, 6), i + 1, dtype="int32")).tofile(
            tmp_path / f"myscan_scan5_{i:04d}.raw")
    src = SpecSource(p, scan=5, image_dir=tmp_path,
                     read_image_kwargs={"detector_shape": (6, 6),
                                        "raw_dtype": "int32"})
    sf = src.frame_for(1)
    assert sf.source_path is not None
    assert sf.source_path.name == "myscan_scan5_0001.raw"

    # metadata-only SPEC (no image_dir) → no raw → source_path stays None
    bare = SpecSource(p, scan=5)
    assert bare.frame_for(1).source_path is None
