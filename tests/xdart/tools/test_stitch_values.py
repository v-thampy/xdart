from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from xdart.gui.tools.stitch_values import (
    StitchFrameSelector,
    StitchToolForm,
    StitchToolPreflightRefused,
    prepare_stitch_tool,
)
from xrd_tools.analysis.module_transaction import MetadataColumnSelector
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.scan_operations import analysis_canonical_fingerprint
from xrd_tools.analysis.stitch_operation import (
    StitchGeometryKind,
    XuStitchOperationPlan,
    run_stitch_operation,
)
from xrd_tools.analysis.xu_stitch_calibration import canonical_surface_resource_bytes
from xrd_tools.io.analysis_artifact import AnalysisArtifactOverwrite


def test_form_is_canonical_fingerprinted_and_keeps_2d_held(stitch_form):
    assert stitch_form.raw_dtype == np.dtype("int32").str
    assert len(stitch_form.fingerprint) == 64
    assert replace(stitch_form, threshold=700_000).fingerprint != stitch_form.fingerprint
    with pytest.raises(ValueError, match="2-D Stitch remains held"):
        replace(stitch_form, mode="2d")

    monitor = stitch_form.monitor_selector
    legacy = (
        "stitch-tool-form-v1",
        stitch_form.project_root,
        stitch_form.spec_path,
        stitch_form.scan,
        stitch_form.image_dir,
        stitch_form.image_stem,
        stitch_form.frame_selector.fingerprint_value,
        stitch_form.detector_shape,
        stitch_form.raw_dtype,
        stitch_form.raw_header_skip,
        stitch_form.threshold,
        stitch_form.geometry_path,
        stitch_form.geometry_kind,
        stitch_form.expected_geometry_sha256,
        stitch_form.source_motors,
        stitch_form.poni_references,
        stitch_form.image_rotation,
        stitch_form.q_range,
        stitch_form.npt_1d,
        None if monitor is None else (monitor.name, monitor.occurrence),
        stitch_form.use_detector_mask,
        stitch_form.output_path,
        stitch_form.overwrite,
        stitch_form.max_frame_bytes,
        stitch_form.mode,
    )
    assert stitch_form.fingerprint == analysis_canonical_fingerprint(
        "stitch-tool-form-v1",
        legacy,
    )


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


def _xu_form(tmp_path: Path) -> StitchToolForm:
    spec = tmp_path / "surface"
    spec.write_text(
        "#F surface\n"
        "#E 1\n"
        "#D today\n"
        "#O0 del  nu\n\n"
        "#S 14 ascan del 14 14 0 1\n"
        "#D today\n"
        "#P0 14 -10\n"
        "#N 5\n"
        "#L del  nu  I0  energy  det\n"
        "14 -10 2 17000.018 1\n",
        encoding="utf-8",
    )
    images = tmp_path / "images"
    images.mkdir()
    np.ones((195, 1475), dtype=np.int32).tofile(
        images / "surface_scan14_0000.raw"
    )
    asset = tmp_path / "calibration" / "xu" / "surface.json"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(canonical_surface_resource_bytes())
    output = tmp_path / "output"
    output.mkdir()
    return StitchToolForm(
        project_root=tmp_path,
        spec_path=spec,
        scan="14",
        image_dir=images,
        image_stem="surface_scan14_",
        frame_selector=StitchFrameSelector(0, 0, 1),
        detector_shape=(195, 1475),
        raw_dtype="int32",
        raw_header_skip=0,
        threshold=800_000,
        geometry_path=asset,
        geometry_kind=StitchGeometryKind.PYFAI_GONIOMETER_JSON,
        source_motors=(("del", "del"), ("nu", "nu")),
        q_range=(1.0, 5.2),
        npt_1d=64,
        monitor_selector=MetadataColumnSelector("I0"),
        use_detector_mask=True,
        output_path=output / "stitched.nexus",
        overwrite=AnalysisArtifactOverwrite.CREATE_NEW,
        max_frame_bytes=4 * 1024 * 1024,
        backend="xu_hist",
    )


def test_xu_form_is_backend_tagged_and_preflight_binds_asset_runtime(tmp_path):
    form = _xu_form(tmp_path)
    preflight = prepare_stitch_tool(form)
    assert form.backend == "xu_hist"
    assert type(preflight.request.plan) is XuStitchOperationPlan
    assert preflight.summary.backend == "xu_hist"
    assert preflight.summary.asset_semantic_fingerprint == (
        preflight.request.plan.calibration.semantic_fingerprint
    )
    assert preflight.summary.effective_geometry_fingerprint == (
        preflight.request.plan.effective_geometry.fingerprint
    )
    assert [selector.name for selector in preflight.summary.required_selectors] == [
        "I0",
        "del",
        "energy",
        "nu",
    ]
    assert preflight.request.provenance["observations"][
        "source_energy_range_eV"
    ] == [17000.018, 17000.018]


def test_xu_form_runs_its_production_spec_raw_request_to_v2(tmp_path):
    preflight = prepare_stitch_tool(_xu_form(tmp_path))
    result = run_stitch_operation(preflight.request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    assert result.terminal.request is preflight.request.module
    assert result.payload.schema_version == 2
    assert result.payload.execution_attestation_digest == (
        result.terminal.commit.execution_attestation_digest
    )
    assert result.payload.coverage is not None
    assert np.any(result.payload.coverage > 0)


@pytest.mark.parametrize(
    "change",
    (
        {"detector_shape": (195, 1474)},
        {"raw_dtype": "uint16"},
        {"threshold": None},
        {"image_rotation": 90},
        {"use_detector_mask": False},
    ),
)
def test_xu_form_refuses_changes_to_asset_locked_fields(tmp_path, change):
    form = _xu_form(tmp_path)
    with pytest.raises(ValueError, match="locked SURFACE"):
        replace(form, **change)
