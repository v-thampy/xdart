from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

import xrd_tools.analysis.xu_stitch_calibration as xu_module
from xrd_tools.analysis.xu_stitch_calibration import (
    CANONICAL_XU_STITCH_CALIBRATION_LOCATOR,
    XuStitchCalibrationInput,
    XuStitchCalibrationRefused,
    canonical_surface_resource_bytes,
    capture_xu_stitch_calibration,
    install_canonical_xu_stitch_calibration,
    parse_xu_stitch_calibration_bytes,
    revalidate_xu_stitch_calibration,
)


EXPECTED_SHA = "57857833c56eeed0db27ec9e3f64aa635d5cd1d1e74eaed0b957eb054b3356d3"
EXPECTED_SEMANTIC = "90f3bec535ed9a21be1b9d93491e774364df5849155b9b9c1707eace848e40f8"


def _symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    """Create *link* -> *target*; skip where the host refuses symbolic links."""
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symbolic links unavailable here: {error}")


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


def test_canonical_surface_installer_is_create_only_and_idempotent(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    first = install_canonical_xu_stitch_calibration(project_root=project)
    target = project / CANONICAL_XU_STITCH_CALIBRATION_LOCATOR
    assert target.read_bytes() == canonical_surface_resource_bytes()
    second = install_canonical_xu_stitch_calibration(project_root=project)
    assert second.fingerprint == first.fingerprint
    assert second.file_state == first.file_state

    conflict_project = tmp_path / "conflict"
    conflict_target = conflict_project / CANONICAL_XU_STITCH_CALIBRATION_LOCATOR
    conflict_target.parent.mkdir(parents=True)
    conflict_target.write_bytes(b"foreign")
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        install_canonical_xu_stitch_calibration(project_root=conflict_project)
    assert raised.value.code == "XU_CALIBRATION_INSTALL_CONFLICT"
    assert conflict_target.read_bytes() == b"foreign"


def test_by_name_installer_installs_readmits_and_conflicts_like_the_descriptor_walk(
    tmp_path, monkeypatch,
):
    """The no-dir_fd installer (Windows' only one) on any host."""
    monkeypatch.setattr(xu_module, "_DESCRIPTOR_WALK", False)
    project = tmp_path / "project"
    project.mkdir()
    first = install_canonical_xu_stitch_calibration(project_root=project)
    target = project / CANONICAL_XU_STITCH_CALIBRATION_LOCATOR
    assert target.read_bytes() == canonical_surface_resource_bytes()
    second = install_canonical_xu_stitch_calibration(project_root=project)
    assert second.fingerprint == first.fingerprint
    assert second.file_state == first.file_state


def _descriptor_walk_modes(module):
    modes = [pytest.param(False, id="by_name")]
    if module._DESCRIPTOR_WALK:
        modes.insert(0, pytest.param(True, id="descriptor"))
    return modes


@pytest.mark.parametrize("descriptor_walk", _descriptor_walk_modes(xu_module))
def test_installer_never_writes_outside_after_a_parent_exchange(
    tmp_path, monkeypatch, descriptor_walk,
):
    """Codex PR #1 review, F2: a parent directory exchanged for a symbolic
    link between the ancestry inspection and the leaf open must not land
    the asset inside the link's target.  The descriptor walk cannot follow
    it; the by-name walk reads the created leaf's final path and refuses,
    removing the misplaced empty leaf."""
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    target = project / CANONICAL_XU_STITCH_CALIBRATION_LOCATOR
    target.parent.mkdir(parents=True)
    parked = target.parent.with_name(target.parent.name + "-parked")
    real_open = os.open
    exchanged = False

    def exchange_then_open(path, flags, *args, **kwargs):
        nonlocal exchanged
        if not exchanged and flags & os.O_CREAT and Path(path).name == target.name:
            exchanged = True
            target.parent.rename(parked)
            _symlink(target.parent, outside, directory=True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(xu_module, "_DESCRIPTOR_WALK", descriptor_walk)
    monkeypatch.setattr(os, "open", exchange_then_open)
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        install_canonical_xu_stitch_calibration(project_root=project)
    assert exchanged
    assert raised.value.code == "XU_CALIBRATION_INSTALL_FAILED"
    assert list(outside.iterdir()) == []
    if descriptor_walk:
        # The held directory descriptor still names the inspected (now
        # parked) directory: the asset lands there, and only the lexical
        # re-admission through the exchanged parent refuses.
        assert [entry.name for entry in parked.iterdir()] == [target.name]
        assert (parked / target.name).read_bytes() == canonical_surface_resource_bytes()
    else:
        assert list(parked.iterdir()) == []


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


@pytest.mark.parametrize("locator", ["surface\x00.json", "surface-\udcff.json"])
def test_surface_input_normalizes_hostile_locator_text(locator):
    with pytest.raises(TypeError, match="XU calibration locator"):
        XuStitchCalibrationInput(locator)


@pytest.mark.parametrize("project_root", ["project\x00root", "project-\udcff-root"])
def test_surface_capture_normalizes_hostile_project_text(project_root):
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        capture_xu_stitch_calibration(
            XuStitchCalibrationInput("surface.json"),
            project_root=project_root,
        )
    assert raised.value.code == "XU_CALIBRATION_PROJECT_INVALID"


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
    _symlink(linked, outside)
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
    _symlink(parent_alias, real_parent, directory=True)
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
    nonregular = ["surface.dir"]
    (project / "surface.dir").mkdir()
    if hasattr(os, "mkfifo"):
        os.mkfifo(project / "surface.fifo")
        nonregular.append("surface.fifo")
    for locator in nonregular:
        with pytest.raises(XuStitchCalibrationRefused) as raised:
            capture_xu_stitch_calibration(
                XuStitchCalibrationInput(locator),
                project_root=project,
            )
        assert raised.value.code == "XU_CALIBRATION_NOT_REGULAR", locator

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
    _symlink(resource, real)
    monkeypatch.setattr(
        "xrd_tools.analysis.xu_stitch_calibration.resources.files",
        lambda _package: package,
    )
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        canonical_surface_resource_bytes()
    assert raised.value.code == "XU_CANONICAL_ASSET_UNAVAILABLE"


@pytest.mark.parametrize("linked_component", ["package", "assets", "xu"])
def test_canonical_resource_refuses_symlink_in_package_ancestry(
    tmp_path,
    monkeypatch,
    linked_component,
):
    raw = canonical_surface_resource_bytes()
    real_package = tmp_path / "real-package"
    resource = (
        real_package / "assets" / "xu" / "psic_powder_1d_surface_v1.json"
    )
    resource.parent.mkdir(parents=True)
    resource.write_bytes(raw)
    if linked_component == "package":
        package = tmp_path / "package"
        _symlink(package, real_package, directory=True)
    else:
        package = tmp_path / "package"
        package.mkdir()
        if linked_component == "assets":
            _symlink(package / "assets", real_package / "assets", directory=True)
        else:
            (package / "assets").mkdir()
            _symlink(
                package / "assets" / "xu",
                real_package / "assets" / "xu",
                directory=True,
            )
    monkeypatch.setattr(
        "xrd_tools.analysis.xu_stitch_calibration.resources.files",
        lambda _package: package,
    )
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        canonical_surface_resource_bytes()
    assert raised.value.code == "XU_CANONICAL_ASSET_UNAVAILABLE"


@pytest.mark.parametrize("package_node", [object(), "package\x00root"])
def test_canonical_resource_refuses_nonfilesystem_or_hostile_root(
    monkeypatch,
    package_node,
):
    monkeypatch.setattr(
        "xrd_tools.analysis.xu_stitch_calibration.resources.files",
        lambda _package: package_node,
    )
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        canonical_surface_resource_bytes()
    assert raised.value.code == "XU_CANONICAL_ASSET_UNAVAILABLE"


def test_canonical_resource_refuses_noncanonical_package_root(
    tmp_path,
    monkeypatch,
):
    real_package = tmp_path / "package"
    real_package.mkdir()
    alias = tmp_path / "alias"
    _symlink(alias, real_package, directory=True)
    hostile = os.path.join(os.fspath(alias), os.pardir, real_package.name)
    monkeypatch.setattr(
        "xrd_tools.analysis.xu_stitch_calibration.resources.files",
        lambda _package: hostile,
    )
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        canonical_surface_resource_bytes()
    assert raised.value.code == "XU_CANONICAL_ASSET_UNAVAILABLE"


def test_exact_historical_xu_json_requires_reviewed_migration():
    legacy = (
        Path(__file__).parent / "fixtures" / "xu_geometry_del_nu_legacy.json"
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
