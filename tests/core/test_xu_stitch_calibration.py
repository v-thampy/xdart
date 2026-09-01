from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from xrd_tools.analysis.xu_stitch_calibration import (
    XuStitchCalibrationInput,
    XuStitchCalibrationRefused,
    canonical_surface_resource_bytes,
    capture_xu_stitch_calibration,
    parse_xu_stitch_calibration_bytes,
    revalidate_xu_stitch_calibration,
)


EXPECTED_SHA = "57857833c56eeed0db27ec9e3f64aa635d5cd1d1e74eaed0b957eb054b3356d3"
EXPECTED_SEMANTIC = "90f3bec535ed9a21be1b9d93491e774364df5849155b9b9c1707eace848e40f8"


def test_canonical_surface_resource_has_authenticated_exact_bytes():
    raw = canonical_surface_resource_bytes()
    projection = parse_xu_stitch_calibration_bytes(raw)
    assert len(raw) == 4_837
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA
    assert projection.raw_sha256 == EXPECTED_SHA
    assert projection.semantic_fingerprint == EXPECTED_SEMANTIC
    assert projection.content == raw
    assert projection.value["preset"] == "psic_powder_1d"
    with pytest.raises(TypeError):
        projection.value["preset"] = "changed"
    with pytest.raises(TypeError):
        projection.value["xrayutilities"]["geometry"] = "changed"
    with pytest.raises(TypeError):
        projection.value["xrayutilities"]["sample_axes"][0] = "changed"
    assert not raw.endswith(b"\n")


def test_surface_projection_and_receipt_factory_claims_cannot_be_reused(tmp_path):
    projection = parse_xu_stitch_calibration_bytes(
        canonical_surface_resource_bytes()
    )
    with pytest.raises(TypeError, match="factory-owned"):
        replace(projection, canonical_json="{}")

    project = tmp_path / "project"
    project.mkdir()
    target = project / "surface.json"
    target.write_bytes(canonical_surface_resource_bytes())
    receipt = capture_xu_stitch_calibration(
        XuStitchCalibrationInput("surface.json"),
        project_root=project,
    )
    with pytest.raises(TypeError, match="factory-owned"):
        replace(receipt, fingerprint="0" * 64)


def test_surface_input_freezes_custom_pathlike_to_owned_text():
    class MutablePath:
        shown = "calibration/xu/surface.json"

        def __fspath__(self):
            return self.shown

    mutable = MutablePath()
    request = XuStitchCalibrationInput(mutable)
    mutable.shown = "changed.json"
    assert request.locator == "calibration/xu/surface.json"
    assert type(request.locator) is str


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda raw: b"\xef\xbb\xbf" + raw, "XU_CALIBRATION_PARSE_FAILED"),
        (lambda raw: raw + b"\n", "XU_CALIBRATION_PARSE_FAILED"),
        (lambda raw: raw + b" ", "XU_CALIBRATION_NOT_CANONICAL"),
        (lambda raw: raw.replace(b'"version":1', b'"version":2'), "XU_CALIBRATION_SCHEMA_UNSUPPORTED"),
        (lambda raw: raw.replace(b'"version":1', b'"version":1,"version":1'), "XU_CALIBRATION_PARSE_FAILED"),
    ],
)
def test_surface_parser_refuses_noncanonical_or_foreign_bytes(mutate, code):
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        parse_xu_stitch_calibration_bytes(mutate(canonical_surface_resource_bytes()))
    assert raised.value.code == code


def test_surface_capture_binds_lexical_resolved_and_physical_identity(tmp_path):
    project = tmp_path / "project"
    target = project / "calibration" / "xu" / "surface.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(canonical_surface_resource_bytes())
    request = XuStitchCalibrationInput("calibration/xu/surface.json")
    receipt = capture_xu_stitch_calibration(request, project_root=project)
    assert receipt.lexical_relative_path == "calibration/xu/surface.json"
    assert receipt.resolved_relative_path == "calibration/xu/surface.json"
    assert receipt.byte_count == 4_837
    assert receipt.raw_sha256 == EXPECTED_SHA
    assert receipt.semantic_fingerprint == EXPECTED_SEMANTIC
    assert revalidate_xu_stitch_calibration(receipt) == receipt.content

    target.touch()
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        revalidate_xu_stitch_calibration(receipt)
    assert raised.value.code == "XU_CALIBRATION_IDENTITY_MISMATCH"


def test_surface_capture_refuses_outside_project_and_symlink(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_bytes(canonical_surface_resource_bytes())
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        capture_xu_stitch_calibration(
            XuStitchCalibrationInput(outside), project_root=project
        )
    assert raised.value.code == "XU_CALIBRATION_OUTSIDE_PROJECT"

    linked = project / "surface.json"
    linked.symlink_to(outside)
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        capture_xu_stitch_calibration(
            XuStitchCalibrationInput(linked), project_root=project
        )
    assert raised.value.code == "XU_CALIBRATION_SYMLINK_REFUSED"

    real_parent = tmp_path / "real-parent"
    real_project = real_parent / "project"
    real_target = real_project / "surface.json"
    real_project.mkdir(parents=True)
    real_target.write_bytes(canonical_surface_resource_bytes())
    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        capture_xu_stitch_calibration(
            XuStitchCalibrationInput("surface.json"),
            project_root=parent_alias / "project",
        )
    assert raised.value.code == "XU_CALIBRATION_SYMLINK_REFUSED"


def test_surface_capture_refuses_nonregular_before_read_and_normalizes_path_error(
    tmp_path,
):
    project = tmp_path / "project"
    project.mkdir()
    fifo = project / "surface.fifo"
    os.mkfifo(fifo)
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        capture_xu_stitch_calibration(
            XuStitchCalibrationInput("surface.fifo"),
            project_root=project,
        )
    assert raised.value.code == "XU_CALIBRATION_NOT_REGULAR"

    regular_parent = project / "regular-parent"
    regular_parent.write_bytes(b"not a directory")
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        capture_xu_stitch_calibration(
            XuStitchCalibrationInput("regular-parent/surface.json"),
            project_root=project,
        )
    assert raised.value.code == "XU_CALIBRATION_UNAVAILABLE"


def test_canonical_resource_refuses_symlink_even_when_bytes_match(
    tmp_path,
    monkeypatch,
):
    raw = canonical_surface_resource_bytes()
    package = tmp_path / "package"
    resource = package / "assets" / "xu" / "psic_powder_1d_surface_v1.json"
    resource.parent.mkdir(parents=True)
    real = tmp_path / "real.json"
    real.write_bytes(raw)
    resource.symlink_to(real)
    monkeypatch.setattr(
        "xrd_tools.analysis.xu_stitch_calibration.resources.files",
        lambda _package: package,
    )
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        canonical_surface_resource_bytes()
    assert raised.value.code == "XU_CANONICAL_ASSET_UNAVAILABLE"


def test_exact_historical_xu_json_requires_reviewed_migration():
    legacy = Path(
        "/Users/vthampy/repos/example_notebooks/Stitching/"
        "xu_geometry_del_nu.json"
    ).read_bytes()
    assert hashlib.sha256(legacy).hexdigest() == (
        "9bb13babeb60475128dd9d14c833cdd8cd85af7510424c1b860b87231d8c0c25"
    )
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        parse_xu_stitch_calibration_bytes(legacy)
    assert raised.value.code == "XU_CALIBRATION_MIGRATION_REQUIRED"


def test_surface_parser_is_engine_light(monkeypatch):
    imported = []
    real_import = __import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"pyFAI", "xrayutilities"}:
            imported.append(name)
            raise AssertionError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    parsed = parse_xu_stitch_calibration_bytes(canonical_surface_resource_bytes())
    assert parsed.semantic_fingerprint == EXPECTED_SEMANTIC
    assert imported == []
