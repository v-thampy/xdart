from __future__ import annotations

import hashlib
import json
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
    assert not raw.endswith(b"\n")


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
