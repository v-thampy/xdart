from __future__ import annotations

import os
from pathlib import Path

import pytest

from xdart.gui.tools.stitch_values import (
    StitchFrameSelector,
    StitchToolForm,
    prepare_stitch_tool,
)
from xrd_tools.analysis.stitch_operation import StitchGeometryKind


_GEOMETRY_SHA256 = (
    "c74676659d821b6460f37e8b42af02767d127263a0079da67fe6e0d291edbdea"
)


def _root():
    configured = os.environ.get("XDART_TEST_DATA")
    if not configured:
        pytest.skip("XDART_TEST_DATA is not configured")
    root = Path(configured) / "stitching"
    if not root.is_dir():
        pytest.skip("authenticated Stitching test data is unavailable")
    return root


@pytest.mark.slow
def test_scan14_tool_preflight_projects_exact_quick_members_without_decoding(
    monkeypatch,
):
    from xrd_tools.io import image as image_api
    from xrd_tools.sources.spec import SpecSource

    def forbidden(*_args, **_kwargs):
        raise AssertionError("operator Preview decoded a detector frame")

    monkeypatch.setattr(SpecSource, "load_frame", forbidden)
    monkeypatch.setattr(image_api, "read_image", forbidden)
    root = _root()
    data = root / "data"
    calibration = data / "calibration"
    form = StitchToolForm(
        project_root=root,
        spec_path=calibration / "LaB6_17keV",
        scan="14.1",
        image_dir=data / "images",
        image_stem="b_thampy_LaB6_17keV_scan14_",
        frame_selector=StitchFrameSelector(0, 285, 19),
        detector_shape=(195, 1475),
        raw_dtype="int32",
        threshold=800_000,
        geometry_path=calibration / "MG_gonio_object_5images.json",
        geometry_kind=StitchGeometryKind.PYFAI_GONIOMETER_JSON,
        expected_geometry_sha256=_GEOMETRY_SHA256,
        source_motors=(("del_angle", "del"), ("nu_angle", "nu")),
        q_range=(1.0, 6.2),
        npt_1d=1500,
        output_path=root / "operator-preflight-only.nexus",
        max_frame_bytes=64 * 1024 * 1024,
    )
    preflight = prepare_stitch_tool(form)
    summary = preflight.summary
    expected = tuple(range(0, 287, 19))
    assert summary.selected_labels == expected
    assert tuple(member.label for member in summary.members) == expected
    assert tuple(member.relative_path for member in summary.members) == tuple(
        f"data/images/b_thampy_LaB6_17keV_scan14_{label:04d}.raw"
        for label in expected
    )
    assert summary.source_relative_path == "data/calibration/LaB6_17keV"
    assert summary.image_directory_relative_path == "data/images"
    assert summary.geometry_relative_path == (
        "data/calibration/MG_gonio_object_5images.json"
    )
    assert summary.geometry_sha256 == _GEOMETRY_SHA256
    assert summary.output_relative_path == "operator-preflight-only.nexus"
    assert summary.request_fingerprint == preflight.request.module.fingerprint
    assert not (root / "operator-preflight-only.nexus").exists()
