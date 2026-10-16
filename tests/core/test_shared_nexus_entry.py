"""Page Entry overrides select actual detector data through the source owners."""

from dataclasses import replace
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_tools.analysis.stitch_sources import (
    StitchSourceSelection, resolve_stitch_sources, stitch_selection_frames,
)
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.sources import open_source
from xrd_tools.sources.point_images import inspect_point_images
from tests.core.test_spec_source import _spec


def _multi_entry(path, images=None):
    if images is None:
        images = np.arange(12, dtype=np.uint32).reshape(3, 2, 2) + 100
    with h5py.File(path, "w") as handle:
        for name, values in (("entry", np.zeros((2 * len(images), *images.shape[1:]), dtype=images.dtype)),
                             ("experiment", images)):
            entry = handle.create_group(name)
            entry.attrs["NX_class"] = "NXentry"
            entry.create_dataset("data/data", data=values)
    return images


def test_explicit_entry_for_calibration_never_falls_back_to_another_group(tmp_path):
    path = tmp_path / 'calibrant.nxs'
    images = _multi_entry(path)
    with pytest.raises(ValueError, match='missing.*not present'):
        open_source(SourceSpec(path, SourceKind.NEXUS_STACK, entry='missing', options={'nexus_entry': 'missing'}))
    source = open_source(SourceSpec(path, SourceKind.NEXUS_STACK, entry='experiment',
                                   options={'nexus_entry': 'experiment'}))
    try:
        np.testing.assert_array_equal(source.load_frame(2), images[2])
    finally:
        source.close()


@pytest.mark.parametrize("kind", ("NeXus", "Directory"))
def test_explicit_nexus_entry_filters_members_reads_pixels_and_preserves_auto(tmp_path, kind):
    path = tmp_path / "scan.nxs"
    images = _multi_entry(path)
    paths = (str(path if kind == "NeXus" else tmp_path),)
    automatic = StitchSourceSelection(kind, paths)
    assert automatic.fingerprint_value == (kind, paths, "", False, "", "")
    discovered = resolve_stitch_sources(automatic, {})
    assert [member.entry for group in discovered.groups for member in group] == ["entry", "experiment"]

    selected = StitchSourceSelection(kind, paths, nexus_entry="experiment")
    assert selected.fingerprint_value != automatic.fingerprint_value
    resolved = resolve_stitch_sources(selected, {})
    member, = (member for group in resolved.groups for member in group)
    assert member.entry == "experiment" and resolved.pending == ()
    source = open_source(member)
    try:
        np.testing.assert_array_equal(source.load_frame(2), images[2])
    finally:
        source.close()
    if kind == "NeXus":
        assert stitch_selection_frames(selected) == (3, (("scan.nxs: 3 frames"),))
    with pytest.raises(ValueError, match="missing.*not present"):
        resolve_stitch_sources(StitchSourceSelection(kind, paths, nexus_entry="missing"), {})


def test_spec_nexus_entry_controls_layout_pixels_and_point_inspection(tmp_path):
    path = _spec(tmp_path)
    master = tmp_path / "myscan_scan005_master.h5"
    images = _multi_entry(master)
    with h5py.File(master, "r+") as handle:
        short = handle.create_group("short")
        short.attrs["NX_class"] = "NXentry"
        short.create_dataset("data/data", data=images[:2])
    # Keep two real external segments to exercise reads crossing the first segment.
    for index, values in enumerate((images[:2], images[2:]), 1):
        shard = tmp_path / f"myscan_scan005_data_{index:06}.h5"
        with h5py.File(shard, "w") as handle:
            handle.create_dataset("frames", data=values)
        with h5py.File(master, "r+") as handle:
            if index == 1:
                del handle["experiment/data/data"]
            handle[f"experiment/data/data_{index:06}"] = h5py.ExternalLink(shard.name, "/frames")
    spec = SourceSpec(path, SourceKind.SPEC, options={
        "scan": "5.1", "image_dir": str(tmp_path), "require_complete_images": True,
        "nexus_entry": "experiment", "read_image_kwargs": {"threshold": 105},
    })
    source = open_source(spec)
    assert source.frame_indices == [0, 1, 2]
    assert source.metadata_for(2)["th"] == 2 and source.metadata_for(2)["chi"] == 5
    assert source.raw_locator_for(2) == (master, 2)
    native = source.load_native_frame(2)
    assert native.dtype == np.uint32
    np.testing.assert_array_equal(native, images[2])
    assert np.isnan(source.load_frame(2)).all()
    short = open_source(SourceSpec(path, SourceKind.SPEC, options={**spec.options,
        "nexus_entry": "short", "require_complete_images": False}))
    assert short.frame_indices == [0, 1]
    np.testing.assert_array_equal(short.load_native_frame(1), images[1])
    assert inspect_point_images(spec) is None
    selection = StitchSourceSelection("Spec", (str(path),), "5", nexus_entry="experiment")
    selected, = resolve_stitch_sources(selection, {}).groups[0]
    assert selected.options["nexus_entry"] == "experiment"
    with pytest.raises(ValueError, match="missing.*not present"):
        open_source(SourceSpec(path, SourceKind.SPEC, options={**spec.options, "nexus_entry": "missing"}))
    from xrd_tools.analysis.scan_operations import (
        AnalysisDisposition, MetadataTablePlan, MetadataTableRequalificationPlan,
        run_metadata_table, run_metadata_table_requalification,
    )
    table = run_metadata_table(MetadataTablePlan(spec))
    assert table.disposition is AnalysisDisposition.COMPLETED, table.diagnostics
    with h5py.File(tmp_path / "myscan_scan005_data_000002.h5", "r+") as handle:
        handle["frames"][0, 0, 0] += 1
    requalified = run_metadata_table_requalification(
        MetadataTableRequalificationPlan(table.receipt, table.table_fingerprint))
    assert requalified.disposition is AnalysisDisposition.REFUSED


def test_entry_survives_stitch_preflight_and_raw_preview(tmp_path):
    from tests.core.test_stitch_operation import _native_form
    from xrd_tools.analysis.stitch_values import prepare_stitch_tool
    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.io.frame_view import read_contributing_frame_view

    form = _native_form(tmp_path, "multigeometry", scans="1")
    images = []
    for path in sorted(Path(form.image_dir).glob("*scan1_*.raw")):
        images.append(np.fromfile(path, dtype=np.int32).reshape(form.detector_shape))
        path.unlink()  # this test's generated inputs only
    images = np.stack(images)
    master = Path(form.image_dir) / "sample_scan1_master.h5"
    _multi_entry(master, images)
    with h5py.File(master, "r+") as handle:
        del handle["experiment/data/data"]
        for index, values in enumerate(images, 1):
            shard = master.with_name(f"sample_scan1_data_{index:06}.h5")
            with h5py.File(shard, "w") as external:
                external.create_dataset("frames", data=values[None])
            handle[f"experiment/data/data_{index:06}"] = h5py.ExternalLink(shard.name, "/frames")
    form = replace(form, source_selection=replace(form.source_selection, nexus_entry="experiment"))
    prepared = prepare_stitch_tool(form)
    assert json.loads(prepared.request.manifest.source_options_json)["nexus_entry"] == "experiment"
    assert prepared.request._preview_members[1]["source_snapshot"]["dataset_path"].startswith("/experiment/")
    result = run_stitch_operation(prepared.request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal.diagnostic
    scan, frame = result.payload.contributing_frames[1]["key"].split("/")
    view = read_contributing_frame_view(prepared.request.module.output.target,
        int(frame.removeprefix("frame_")), scan=scan.removeprefix("scan_"), full_raw=True)
    np.testing.assert_array_equal(view.raw, images[1])


def test_entry_survives_rsm_member_projection_and_preflight(tmp_path):
    from tests.core.test_rsm_sources import _pair, _groups
    from tests.core.test_rsm_operation_r2 import _form
    from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2
    from xrd_tools.analysis.scan_operations import requalified_analysis_source

    path = _pair(tmp_path / "data", with_images=(), scans=(1,))
    images = np.arange(2 * 195 * 487, dtype=np.uint16).reshape(2, 195, 487)
    _multi_entry(path.parent / "RSM_pair_scan1_master.h5", images)
    selection = StitchSourceSelection("Spec", (str(path),), "1", nexus_entry="experiment")
    (members,), skipped = _groups(selection)
    assert skipped == () and members[0].nexus_entry == "experiment"
    assert members[0].fingerprint != replace(members[0], nexus_entry="").fingerprint
    prepared = prepare_rsm_tool_v2(_form(tmp_path, members))
    source = prepared.request.module.source.members[0]
    assert source.analysis.source_spec.options["nexus_entry"] == "experiment"
    with requalified_analysis_source(source.analysis) as opened:
        np.testing.assert_array_equal(opened.load_frame(1), images[1])
