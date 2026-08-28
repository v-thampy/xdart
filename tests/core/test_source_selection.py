"""Typed mode-specific source-selection contracts."""

import pytest

from xrd_tools.core.scan import SourceKind
from xrd_tools.sources import image_series_spec, open_source
from xrd_tools.sources.selection import (
    DirectorySourceSpec,
    normalize_metadata_format,
    single_image_spec,
)


def test_image_series_spec_freezes_all_members_from_selected_fourth(tmp_path):
    paths = tuple(tmp_path / f"series_{index:04d}.tif" for index in range(1, 6))
    for path in paths:
        path.write_bytes(b"x")
    (tmp_path / "other_0001.tif").write_bytes(b"x")

    spec = image_series_spec(paths[3])

    assert spec.kind is SourceKind.TIFF_SERIES
    assert spec.options["selected_file"] == str(paths[3])
    assert tuple(spec.options["files"]) == tuple(str(path) for path in paths)
    assert spec.options["metadata_format"] == "auto"


def test_container_image_selection_carries_actual_auto_metadata_policy(tmp_path):
    selected = tmp_path / "scan.nxs"

    spec = image_series_spec(selected)

    assert spec.kind is SourceKind.NEXUS_STACK
    assert spec.options["metadata_format"] == "auto"


def test_live_metadata_none_remains_explicitly_off(tmp_path):
    assert normalize_metadata_format(None) is None
    assert normalize_metadata_format("None") is None
    assert DirectorySourceSpec(
        tmp_path,
        metadata_format="None",
    ).metadata_format is None


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


def test_open_source_consumes_explicit_single_image_marker(tmp_path):
    selected = tmp_path / "single_0002.tif"
    selected.write_bytes(b"x")

    source = open_source(single_image_spec(selected))

    assert source.frame_indices == [1]
    assert tuple(source.files) == (selected,)
    assert source.name == "single_0002"


@pytest.mark.parametrize("suffix", (".h5", ".hdf5", ".nxs", ".cxi"))
def test_single_image_rejects_container_extensions(suffix, tmp_path):
    with pytest.raises(ValueError, match="Image Series"):
        single_image_spec(tmp_path / f"container{suffix}")
