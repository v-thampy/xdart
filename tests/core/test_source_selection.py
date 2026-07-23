"""Typed mode-specific source-selection contracts."""

from xrd_tools.core.scan import SourceKind
from xrd_tools.sources import image_series_spec, open_source


def test_image_series_spec_freezes_all_members_from_selected_fourth(tmp_path):
    paths = tuple(tmp_path / f"series_{index:04d}.tif" for index in range(1, 6))
    for path in paths:
        path.write_bytes(b"x")
    (tmp_path / "other_0001.tif").write_bytes(b"x")

    spec = image_series_spec(paths[3])

    assert spec.kind is SourceKind.TIFF_SERIES
    assert spec.options["selected_file"] == str(paths[3])
    assert tuple(spec.options["files"]) == tuple(str(path) for path in paths)


def test_open_source_consumes_frozen_image_series_membership(tmp_path):
    paths = tuple(tmp_path / f"series_{index:04d}.tif" for index in range(1, 6))
    for path in paths:
        path.write_bytes(b"x")
    spec = image_series_spec(paths[3])

    # A later sibling is outside the frozen Run membership.
    (tmp_path / "series_0006.tif").write_bytes(b"x")
    source = open_source(spec)

    assert source.name == "series"
    assert source.frame_indices == [1, 2, 3, 4, 5]
    assert tuple(source.files) == paths
