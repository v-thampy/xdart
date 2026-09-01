"""Metadata-only + image-backed SPEC FrameSource: content-based (extensionless)
classification, ALL per-point + constant motors recorded, scan selection, and
optional raw-image loading via the io.image layer."""
import numpy as np
import pytest

pytest.importorskip("silx")

from xrd_tools.core.scan import SourceSpec
from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition,
    MetadataTablePlan,
    MetadataTableRequalificationPlan,
    run_metadata_table,
    run_metadata_table_requalification,
)
from xrd_tools.io.spec import (
    get_spec_scanned_axes,
    is_spec_file,
    read_spec_scan_columns,
)
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


def test_spec_exposes_declared_scanned_axes_not_counter_columns(tmp_path):
    src = open_source(_spec(tmp_path), scan=5)
    assert src.scanned_axes == ("th",)
    assert "i0" in src.motors
    assert "i0" not in src.scanned_axes


def test_get_spec_scanned_axes_parses_common_commands(tmp_path):
    p = tmp_path / "axis_commands"
    p.write_text("""#F axis_commands
#E 1

#S 1 mesh eta 0 1 1 chi 2 4 1 0.1
#N 3
#L eta  chi  det
0 2 100
1 2 110
0 4 120
1 4 130

#S 2 hklscan 1 1 -1 -1 2.9 3.1 2 0.1
#N 4
#L H  K  L  det
1 -1 2.9 10
1 -1 3.0 11
1 -1 3.1 12

#S 3 loopscan 2 0.1
#N 2
#L Epoch  det
1 10
2 11
""")
    assert get_spec_scanned_axes(p, "1.1") == ("eta", "chi")
    assert get_spec_scanned_axes(p, "2.1") == ("L",)
    assert get_spec_scanned_axes(p, "3.1") == ()


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


def test_spec_occurrence_projection_preserves_duplicate_columns_and_fixed_motors(
    tmp_path,
):
    path = tmp_path / "duplicate_columns"
    path.write_text(
        """#F duplicate_columns
#E 1
#O0 eta  chi  mu

#S 1 ascan eta 0 1 1 1
#P0 0 5 6
#N 4
#L eta  Seconds  Seconds  foil status
0 1 10 101
1 2 20 110
"""
    )

    physical, fixed, npts = read_spec_scan_columns(path, "1.1")
    assert npts == 2
    assert [name for name, _values in physical] == [
        "eta",
        "Seconds",
        "Seconds",
        "foil status",
    ]
    np.testing.assert_allclose(physical[1][1], [1, 2])
    np.testing.assert_allclose(physical[2][1], [10, 20])
    assert fixed == (("eta", 0.0), ("chi", 5.0), ("mu", 6.0))

    source = SpecSource(path, scan=1)
    occurrence_names = [
        name for name, _values in source.physical_metadata_columns
    ]
    assert occurrence_names == [
        "eta",
        "Seconds",
        "Seconds",
        "foil status",
    ]
    assert source.fixed_motor_positions == fixed
    assert "chi" not in source.motors
    np.testing.assert_allclose(source.motor_series("eta"), [0, 1])
    np.testing.assert_allclose(source.motor_series("chi"), [5, 5])
    np.testing.assert_allclose(source.motor_series("mu"), [6, 6])

    table = run_metadata_table(
        MetadataTablePlan(
            path,
            kind=SourceKind.SPEC,
            scan="1",
            column_projection=(
                ("eta", 0),
                ("Seconds", 1),
                ("foil status", 0),
                ("chi", 0),
                ("mu", 0),
            ),
        )
    )
    assert table.disposition is AnalysisDisposition.COMPLETED
    assert [column.name for column in table.columns] == [
        "frame_index",
        "eta",
        "Seconds",
        "Seconds",
        "foil status",
        "chi",
        "mu",
    ]
    seconds = [
        column.numeric for column in table.columns if column.name == "Seconds"
    ]
    np.testing.assert_allclose(seconds[0], [1, 2])
    np.testing.assert_allclose(seconds[1], [10, 20])

    requalified = run_metadata_table_requalification(
        MetadataTableRequalificationPlan(table.receipt, table.table_fingerprint)
    )
    assert requalified.disposition is AnalysisDisposition.COMPLETED
    assert requalified.table_fingerprint == table.table_fingerprint

    missing_occurrence = run_metadata_table(
        MetadataTablePlan(
            path,
            kind=SourceKind.SPEC,
            scan="1",
            column_projection=(("Seconds", 2),),
        )
    )
    assert missing_occurrence.disposition is AnalysisDisposition.REFUSED
    assert missing_occurrence.code == "METADATA_COLUMN_PROJECTION_INVALID"


@pytest.mark.parametrize(
    "projection",
    (
        (("Seconds", 1), ("Seconds", 1)),
        (("Seconds", True),),
        ((" Seconds", 0),),
        (["Seconds", 0],),
        [("Seconds", 0)],
        None,
    ),
)
def test_embedded_spec_projection_rejects_noncanonical_selectors(
    tmp_path,
    projection,
):
    source = SourceSpec(
        tmp_path / "source",
        SourceKind.SPEC,
        options={
            "scan": "1.1",
            "metadata_column_projection": projection,
        },
    )
    with pytest.raises(ValueError):
        MetadataTablePlan(source)


def test_embedded_spec_projection_respects_metadata_table_column_bound(
    tmp_path,
):
    allowed = tuple((f"motor_{index}", 0) for index in range(127))
    source = SourceSpec(
        tmp_path / "source",
        SourceKind.SPEC,
        options={
            "scan": "1.1",
            "metadata_column_projection": allowed,
        },
    )
    plan = MetadataTablePlan(source)
    assert dict(plan.source.options)["metadata_column_projection"] == allowed

    too_many = tuple((f"motor_{index}", 0) for index in range(128))
    source = SourceSpec(
        tmp_path / "source",
        SourceKind.SPEC,
        options={
            "scan": "1.1",
            "metadata_column_projection": too_many,
        },
    )
    with pytest.raises(ValueError, match="exceeds the table column bound"):
        MetadataTablePlan(source)
