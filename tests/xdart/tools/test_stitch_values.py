from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from xdart.gui.tools.stitch_values import (
    StitchFrameSelector,
    StitchToolForm,
    StitchToolPreflightRefused,
    prepare_stitch_tool,
)


def test_form_is_canonical_fingerprinted_and_keeps_2d_held(stitch_form):
    assert stitch_form.raw_dtype == np.dtype("int32").str
    assert len(stitch_form.fingerprint) == 64
    assert replace(stitch_form, threshold=700_000).fingerprint != stitch_form.fingerprint
    with pytest.raises(ValueError, match="2-D Stitch remains held"):
        replace(stitch_form, mode="2d")


@pytest.mark.parametrize("field", ("project_root", "image_dir"))
def test_form_refuses_blank_explicit_paths(stitch_form, field):
    with pytest.raises(ValueError, match="required"):
        replace(stitch_form, **{field: ""})


def test_frame_selector_bounds_before_materializing_huge_range():
    selector = StitchFrameSelector(0, 10**15, 1)
    with pytest.raises(StitchToolPreflightRefused) as refused:
        selector.select((0, 1, 2))
    assert refused.value.code == "FRAME_SELECTION_LIMIT_EXCEEDED"


def test_frame_selector_uses_labels_not_table_ordinals():
    assert StitchFrameSelector(10, 30, 20).select((10, 20, 30)) == (10, 30)


def test_preflight_binds_exact_frame_labels_members_and_request(stitch_form):
    preflight = prepare_stitch_tool(stitch_form)
    summary = preflight.summary
    assert preflight.is_current(stitch_form)
    assert not preflight.is_current(replace(stitch_form, npt_1d=97))
    assert summary.source_scan == "5.1"
    assert summary.selected_labels == (0, 2)
    assert [member.label for member in summary.members] == [0, 2]
    assert [member.relative_path for member in summary.members] == [
        "images/myscan_scan5_0000.raw",
        "images/myscan_scan5_0002.raw",
    ]
    assert [member.source_frame_index for member in summary.members] == [0, 0]
    assert summary.request_fingerprint == preflight.request.module.fingerprint
    assert summary.manifest_fingerprint == preflight.request.manifest.fingerprint
    assert summary.geometry_sha256 == stitch_form.expected_geometry_sha256
    assert summary.image_directory_relative_path == "images"
    assert summary.image_stem == "myscan_scan5_"
    assert summary.detector_shape == (4, 5)
    assert summary.raw_dtype == np.dtype("int32").str
    assert summary.raw_header_skip == 0
    assert summary.threshold == 800_000.0
    assert summary.source_motors == (
        ("del_angle", "del"),
        ("nu_angle", "nu"),
    )
    assert summary.image_rotation == 0
    assert summary.monitor_selector is not None
    assert summary.monitor_selector.name == "I0"
    assert summary.use_detector_mask is True
    assert summary.output_relative_path == "output/stitched.nexus"
    assert summary.holds == (
        "2d-orientation-parity",
        "gi-and-custom-corrections",
    )
    assert len(summary.dependency_files) == 4
    assert len(summary.fingerprint) == 64


def test_preflight_never_decodes_detector_frames(monkeypatch, stitch_form):
    from xrd_tools.io import image as image_api
    from xrd_tools.sources.spec import SpecSource

    def forbidden(*_args, **_kwargs):
        raise AssertionError("preflight decoded a detector frame")

    monkeypatch.setattr(SpecSource, "load_frame", forbidden)
    monkeypatch.setattr(image_api, "read_image", forbidden)
    preflight = prepare_stitch_tool(stitch_form)
    assert preflight.summary.selected_labels == (0, 2)


def test_preflight_refuses_output_outside_project(stitch_form, tmp_path):
    outside = tmp_path.parent / "outside-stitch.nexus"
    with pytest.raises(StitchToolPreflightRefused) as refused:
        prepare_stitch_tool(replace(stitch_form, output_path=outside))
    assert refused.value.code == "OUTPUT_OUTSIDE_PROJECT"


def test_preflight_refuses_missing_output_parent(stitch_form, tmp_path):
    missing = tmp_path / "missing" / "stitched.nexus"
    with pytest.raises(StitchToolPreflightRefused) as refused:
        prepare_stitch_tool(replace(stitch_form, output_path=missing))
    assert refused.value.code == "OUTPUT_PARENT_UNAVAILABLE"
