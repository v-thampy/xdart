from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import xrd_tools.analysis.xu_stitch_calibration as xu_module
from xrd_tools.io import descriptor_path
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


# Only Windows can delete a file through the handle that created it; every
# POSIX host leaves the misplaced (empty) leaf and names it in the refusal.
_DISPOSES_BY_HANDLE = sys.platform == "win32"


def _stage_parent_exchange(monkeypatch, target, outside, *, restore=False):
    """Exchange *target*'s parent for a link to *outside* inside the leaf
    create -- once, whichever installer creates it -- and with *restore*
    put the real parent back before the create returns, so the misplaced
    leaf is reachable only by its final path.  Returns the parked real
    parent and a flag record."""
    parked = target.parent.with_name(target.parent.name + "-parked")
    real_open = os.open
    real_create = descriptor_path.create_exclusive_leaf
    state = {"exchanged": False}

    def exchange():
        state["exchanged"] = True
        target.parent.rename(parked)
        _symlink(target.parent, outside, directory=True)

    def restore_parent():
        os.unlink(target.parent)
        parked.rename(target.parent)

    def exchange_then_open(path, flags, *args, **kwargs):
        if state["exchanged"] or not flags & os.O_CREAT or Path(path).name != target.name:
            return real_open(path, flags, *args, **kwargs)
        exchange()
        descriptor = real_open(path, flags, *args, **kwargs)
        if restore:
            restore_parent()
        return descriptor

    def exchange_then_create(path):
        if state["exchanged"] or Path(path).name != target.name:
            return real_create(path)
        exchange()
        descriptor = real_create(path)
        if restore:
            restore_parent()
        return descriptor

    monkeypatch.setattr(os, "open", exchange_then_open)
    monkeypatch.setattr(descriptor_path, "create_exclusive_leaf", exchange_then_create)
    return parked, state


def _assert_misplaced_leaf_disposition(raised, outside, name):
    """After a by-name refusal: nothing outside where the handle could
    delete the leaf, otherwise exactly the empty leaf, named in the cause."""
    if _DISPOSES_BY_HANDLE:
        assert list(outside.iterdir()) == []
        return
    assert [(entry.name, entry.stat().st_size) for entry in outside.iterdir()] == [
        (name, 0)
    ]
    cause = str(raised.value.__cause__)
    assert os.path.realpath(outside / name) in cause
    assert "empty leaf left there" in cause


@pytest.mark.parametrize("descriptor_walk", _descriptor_walk_modes(xu_module))
def test_installer_never_writes_outside_after_a_parent_exchange(
    tmp_path, monkeypatch, descriptor_walk,
):
    """Codex PR #1 review, F2: a parent directory exchanged for a symbolic
    link between the ancestry inspection and the leaf open must not land
    the asset inside the link's target.  The descriptor walk cannot follow
    it; the by-name walk reads the created leaf's final path and refuses,
    disposing of the misplaced empty leaf through its own handle where
    the platform can, and naming it where it cannot."""
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    target = project / CANONICAL_XU_STITCH_CALIBRATION_LOCATOR
    target.parent.mkdir(parents=True)
    parked, state = _stage_parent_exchange(monkeypatch, target, outside)

    monkeypatch.setattr(xu_module, "_DESCRIPTOR_WALK", descriptor_walk)
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        install_canonical_xu_stitch_calibration(project_root=project)
    assert state["exchanged"]
    assert raised.value.code == "XU_CALIBRATION_INSTALL_FAILED"
    if descriptor_walk:
        # The held directory descriptor still names the inspected (now
        # parked) directory: the asset lands there, and only the lexical
        # re-admission through the exchanged parent refuses.
        assert list(outside.iterdir()) == []
        assert [entry.name for entry in parked.iterdir()] == [target.name]
        assert (parked / target.name).read_bytes() == canonical_surface_resource_bytes()
    else:
        assert list(parked.iterdir()) == []
        _assert_misplaced_leaf_disposition(raised, outside, target.name)


def test_by_name_installer_disposes_of_a_leaf_the_name_no_longer_reaches(
    tmp_path, monkeypatch,
):
    """Codex review of 0ed7a46c, F2: the parent is exchanged for the
    create and put back before the placement check, so the misplaced leaf
    is no longer what the target name resolves to.  A by-name cleanup
    would miss it (or hit whatever now sits at the name); disposal goes
    through the handle, and the project itself is untouched."""
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    target = project / CANONICAL_XU_STITCH_CALIBRATION_LOCATOR
    target.parent.mkdir(parents=True)
    _, state = _stage_parent_exchange(monkeypatch, target, outside, restore=True)

    monkeypatch.setattr(xu_module, "_DESCRIPTOR_WALK", False)
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        install_canonical_xu_stitch_calibration(project_root=project)
    assert state["exchanged"]
    assert raised.value.code == "XU_CALIBRATION_INSTALL_FAILED"
    assert not target.parent.is_symlink()
    assert list(target.parent.iterdir()) == []
    _assert_misplaced_leaf_disposition(raised, outside, target.name)
    # The restored project installs normally afterwards.
    receipt = install_canonical_xu_stitch_calibration(project_root=project)
    assert receipt.content == canonical_surface_resource_bytes()
    assert target.read_bytes() == canonical_surface_resource_bytes()


def test_by_name_installer_never_deletes_a_foreign_file_at_the_misplaced_name(
    tmp_path, monkeypatch,
):
    """Codex review of 0ed7a46c, F2: a foreign file moved over the
    misplaced leaf's name between the placement check and the cleanup
    must survive.  The cleanup holds the leaf's own handle and never
    unlinks a name; where the handle blocks the move (Windows) the foreign
    file simply stays where it was."""
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    target = project / CANONICAL_XU_STITCH_CALIBRATION_LOCATOR
    target.parent.mkdir(parents=True)
    _stage_parent_exchange(monkeypatch, target, outside)
    payload = b"foreign payload must survive"
    foreign = outside / "foreign.tmp"
    outsider = outside / target.name
    foreign.write_bytes(payload)
    real_dispose = descriptor_path.dispose_created_leaf
    moved = {"replaced": None}

    def replace_then_dispose(descriptor):
        assert moved["replaced"] is None
        try:
            os.replace(foreign, outsider)
        except OSError:
            moved["replaced"] = False
        else:
            moved["replaced"] = True
        return real_dispose(descriptor)

    monkeypatch.setattr(descriptor_path, "dispose_created_leaf", replace_then_dispose)
    monkeypatch.setattr(xu_module, "_DESCRIPTOR_WALK", False)
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        install_canonical_xu_stitch_calibration(project_root=project)
    assert raised.value.code == "XU_CALIBRATION_INSTALL_FAILED"
    assert moved["replaced"] is not None
    survivor = outsider if moved["replaced"] else foreign
    assert survivor.read_bytes() == payload
    if _DISPOSES_BY_HANDLE:
        assert [entry.name for entry in outside.iterdir()] == [survivor.name]
    else:
        # POSIX replaces the open leaf by name; the foreign file now holds it.
        assert moved["replaced"]
        assert [entry.name for entry in outside.iterdir()] == [outsider.name]


@pytest.mark.parametrize("descriptor_walk", _descriptor_walk_modes(xu_module))
def test_installer_refuses_a_link_planted_at_the_leaf_name(
    tmp_path, monkeypatch, descriptor_walk,
):
    """A symbolic link planted at the leaf name between the existence
    check and the create is an existing entry, never a path to create
    through: the exclusive create conflicts and the link's target is not
    made.  (POSIX O_EXCL; on Windows CREATE_NEW opens the reparse point.)"""
    project = tmp_path / "project"
    project.mkdir()
    target = project / CANONICAL_XU_STITCH_CALIBRATION_LOCATOR
    target.parent.mkdir(parents=True)
    sibling = target.parent / "sibling.json"
    real_open = os.open
    real_create = descriptor_path.create_exclusive_leaf
    planted = {"done": False}

    def plant():
        if not planted["done"]:
            planted["done"] = True
            _symlink(target, sibling)

    def plant_then_open(path, flags, *args, **kwargs):
        if flags & os.O_CREAT and Path(path).name == target.name:
            plant()
        return real_open(path, flags, *args, **kwargs)

    def plant_then_create(path):
        plant()
        return real_create(path)

    monkeypatch.setattr(os, "open", plant_then_open)
    monkeypatch.setattr(descriptor_path, "create_exclusive_leaf", plant_then_create)
    monkeypatch.setattr(xu_module, "_DESCRIPTOR_WALK", descriptor_walk)
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        install_canonical_xu_stitch_calibration(project_root=project)
    assert planted["done"]
    assert not sibling.exists()
    assert target.is_symlink()
    assert raised.value.code == "XU_CALIBRATION_INSTALL_CONFLICT"


def test_by_name_installer_disposes_of_a_leaf_it_failed_to_fill_through_the_handle(
    tmp_path, monkeypatch,
):
    """A write failure after a verified placement disposes of the leaf
    through its handle (Windows) or leaves it and says so (POSIX); a name
    is never unlinked."""
    project = tmp_path / "project"
    project.mkdir()
    target = project / CANONICAL_XU_STITCH_CALIBRATION_LOCATOR
    attacked = {"done": False}

    def fail_write(_descriptor, _payload):
        assert not attacked["done"]
        attacked["done"] = True
        raise OSError("synthetic write failure")

    monkeypatch.setattr(xu_module.os, "write", fail_write)
    monkeypatch.setattr(xu_module, "_DESCRIPTOR_WALK", False)
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        install_canonical_xu_stitch_calibration(project_root=project)
    assert attacked["done"]
    assert raised.value.code == "XU_CALIBRATION_INSTALL_FAILED"
    cause = raised.value.__cause__
    assert str(cause) == "synthetic write failure"
    if _DISPOSES_BY_HANDLE:
        assert not target.exists()
        assert not getattr(cause, "__notes__", [])
    else:
        assert target.stat().st_size == 0
        assert cause.__notes__ == [f"partial canonical asset left at {target}"]


@pytest.mark.skipif(
    sys.platform == "win32", reason="an open file cannot be replaced on Windows"
)
def test_by_name_installer_never_unlinks_a_foreign_replacement_after_a_failed_fill(
    tmp_path, monkeypatch,
):
    project = tmp_path / "project"
    project.mkdir()
    target = project / CANONICAL_XU_STITCH_CALIBRATION_LOCATOR
    foreign = b"foreign replacement"
    attacked = {"done": False}

    def replace_name_then_fail(_descriptor, _payload):
        assert not attacked["done"]
        attacked["done"] = True
        target.unlink()
        target.write_bytes(foreign)
        raise OSError("synthetic write failure")

    monkeypatch.setattr(xu_module.os, "write", replace_name_then_fail)
    monkeypatch.setattr(xu_module, "_DESCRIPTOR_WALK", False)
    with pytest.raises(XuStitchCalibrationRefused) as raised:
        install_canonical_xu_stitch_calibration(project_root=project)
    assert attacked["done"]
    assert raised.value.code == "XU_CALIBRATION_INSTALL_FAILED"
    assert target.read_bytes() == foreign


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
