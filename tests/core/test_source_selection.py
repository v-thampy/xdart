"""Typed mode-specific source-selection contracts."""

import os

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


def test_image_series_spec_checks_file_type_only_for_series_names(
    tmp_path, monkeypatch,
):
    """A shared acquisition folder: an unrelated name gets no metadata call."""

    for name in (
        "series_2.tif", "series_02.tif", "SERIES_0003.TIF", "series_0004.tif",
        "series-0005.tif", "series_10.tif",
        "other_0001.tif", "series_0001.cbf", "series_0001.tif.bak",
        "series.tif", "series_x.tif", "seriesB_0001.tif", "notes.txt",
    ):
        (tmp_path / name).write_bytes(b"x")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "frame.tif").write_bytes(b"x")
    (tmp_path / "series_0011.tif").symlink_to(elsewhere / "frame.tif")
    (tmp_path / "series_0012.tif").mkdir()
    (tmp_path / "series_0013.tif").symlink_to(tmp_path / "missing.tif")
    (tmp_path / "series_0014.tif").symlink_to(elsewhere)
    (tmp_path / "other_link").symlink_to(tmp_path / "nowhere")
    checked = []
    for name in ("stat", "lstat"):
        call = getattr(os, name)

        def observed(path, *args, _call=call, **kwargs):
            checked.append(path)
            return _call(path, *args, **kwargs)

        monkeypatch.setattr(os, name, observed)

    spec = image_series_spec(tmp_path / "series_0004.tif")
    monkeypatch.undo()

    assert tuple(spec.options["files"]) == tuple(
        str(tmp_path / name)
        for name in (
            "series_02.tif", "series_2.tif", "SERIES_0003.TIF",
            "series_0004.tif", "series-0005.tif", "series_10.tif",
            "series_0011.tif",
        )
    )
    assert spec.options["pattern"] == "series_*.tif"
    assert spec.options["scan_name"] == "series"
    assert checked
    assert {os.path.basename(path) for path in checked} <= {
        "series_2.tif", "series_02.tif", "SERIES_0003.TIF", "series_0004.tif",
        "series-0005.tif", "series_10.tif", "series_0011.tif",
        "series_0012.tif", "series_0013.tif", "series_0014.tif",
    }


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
