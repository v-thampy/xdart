from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import stat

import pytest

from xrd_tools.analysis.rsm_geometry_asset import (
    CANONICAL_RSM_GEOMETRY_LOCATOR,
    RSMGeometryAssetInput,
    RSMGeometryAssetRefused,
    canonical_rsm_geometry_resource_bytes,
    capture_rsm_geometry_asset,
    install_canonical_rsm_geometry_asset,
    parse_rsm_geometry_asset_bytes,
    revalidate_rsm_geometry_asset,
)


EXPECTED_SIZE = 745
EXPECTED_SHA256 = (
    "0f22b00363ff93fec7b97c7b5c31f8b8e389aac3aae334cbadbc213b629d9f84"
)
EXPECTED_SEMANTIC = (
    "a5841542c58c5ff33899ef13a6f3883c6f6e356eefb11c41d1f24936eed9f0fa"
)


def _canonical_value() -> dict[str, object]:
    return json.loads(canonical_rsm_geometry_resource_bytes())


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def test_canonical_rsm_resource_has_frozen_exact_identity_and_values():
    raw = canonical_rsm_geometry_resource_bytes()
    projection = parse_rsm_geometry_asset_bytes(raw)
    assert len(raw) == EXPECTED_SIZE
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256
    assert projection.byte_count == EXPECTED_SIZE
    assert projection.raw_sha256 == EXPECTED_SHA256
    assert projection.semantic_fingerprint == EXPECTED_SEMANTIC
    assert projection.content == raw
    assert not raw.endswith(b"\n")

    value = projection.value
    assert value["schema"] == "xdart.rsm_geometry"
    assert value["version"] == 1
    assert value["preset"] == "ssrl17_2_psic_pilatus300k_sto"
    assert value["diffractometer"] == {
        "camera": ("z-", "x-"),
        "detector_circles": ("x+", "z-"),
        "hxrd_geometry": "real",
        "hxrd_n": (0.0, 1.0, 0.0),
        "hxrd_q": (0.0, 0.0, 1.0),
        "motor_roles": ("mu", "eta", "chi", "phi", "nu", "del"),
        "preset": "psic",
        "r_i": (0.0, 1.0, 0.0),
        "sample_circles": ("x+", "z-", "y+", "z-"),
    }
    assert value["detector"]["header"] == {
        "Nch1": 195,
        "Nch2": 487,
        "cch1": 97.0,
        "cch2": 243.0,
        "distance": 1014.7173,
        "pwidth1": 0.172,
        "pwidth2": 0.172,
    }
    assert value["detector"]["image_orientation"] == {
        "flip_horizontal": False,
        "flip_vertical": False,
        "rotation": 0,
        "transpose": False,
    }
    assert value["detector"]["roi"] == (0, -1, 0, -1)
    assert value["validation"] == {
        "notebook": "RSM_process.ipynb",
        "notebook_sha256": (
            "980d6caae252b86cfd68e7cc605621f16a539f341b8abdb2ab946d43d84808ac"
        ),
        "reference_scan": "43.1",
    }
    with pytest.raises(TypeError):
        value["preset"] = "changed"
    with pytest.raises(TypeError):
        value["diffractometer"]["camera"][0] = "x+"


@pytest.mark.parametrize(
    "mutation,code",
    [
        (lambda raw: b"\xef\xbb\xbf" + raw, "RSM_GEOMETRY_PARSE_FAILED"),
        (lambda raw: raw + b"\x00", "RSM_GEOMETRY_PARSE_FAILED"),
        (lambda raw: raw + b"\n", "RSM_GEOMETRY_PARSE_FAILED"),
        (lambda raw: raw + b" ", "RSM_GEOMETRY_NOT_CANONICAL"),
        (
            lambda raw: raw.replace(b'"version":1', b'"version":1,"version":1'),
            "RSM_GEOMETRY_PARSE_FAILED",
        ),
        (
            lambda raw: raw.replace(b'"version":1', b'"version":2'),
            "RSM_GEOMETRY_SCHEMA_UNSUPPORTED",
        ),
        (
            lambda raw: raw.replace(b'"version":1', b'"unknown":0,"version":1'),
            "RSM_GEOMETRY_SCHEMA_UNSUPPORTED",
        ),
        (
            lambda raw: raw.replace(b'"rotation":0', b'"rotation":NaN'),
            "RSM_GEOMETRY_PARSE_FAILED",
        ),
        (
            lambda raw: raw.replace(b'"rotation":0', b'"rotation":"\\udcff"'),
            "RSM_GEOMETRY_PARSE_FAILED",
        ),
    ],
)
def test_rsm_parser_refuses_hostile_noncanonical_and_foreign_bytes(mutation, code):
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        parse_rsm_geometry_asset_bytes(mutation(canonical_rsm_geometry_resource_bytes()))
    assert raised.value.code == code


def test_every_rsm_asset_leaf_participates_in_authenticated_identity():
    def leaf_paths(value, prefix=()):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from leaf_paths(child, prefix + (key,))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from leaf_paths(child, prefix + (index,))
        else:
            yield prefix

    original = _canonical_value()
    paths = tuple(leaf_paths(original))
    assert len(paths) == 46
    for path in paths:
        changed = copy.deepcopy(original)
        owner = changed
        for part in path[:-1]:
            owner = owner[part]
        current = owner[path[-1]]
        if type(current) is bool:
            replacement = not current
        elif type(current) is int:
            replacement = current + 1
        elif type(current) is float:
            replacement = current + 0.5
        else:
            replacement = current + "-changed"
        owner[path[-1]] = replacement
        with pytest.raises(RSMGeometryAssetRefused) as raised:
            parse_rsm_geometry_asset_bytes(_canonical(changed))
        assert raised.value.code == "RSM_GEOMETRY_SCHEMA_UNSUPPORTED"


def test_rsm_projection_and_receipt_are_factory_owned_noncopyable(tmp_path):
    projection = parse_rsm_geometry_asset_bytes(canonical_rsm_geometry_resource_bytes())
    for operation in (
        copy.copy,
        copy.deepcopy,
        lambda value: pickle.dumps(value),
        lambda value: replace(value, canonical_json="{}"),
    ):
        with pytest.raises(TypeError):
            operation(projection)

    project = tmp_path / "project"
    project.mkdir()
    target = project / "geometry.json"
    target.write_bytes(canonical_rsm_geometry_resource_bytes())
    receipt = capture_rsm_geometry_asset(
        RSMGeometryAssetInput("geometry.json"), project_root=project
    )
    for operation in (
        copy.copy,
        copy.deepcopy,
        lambda value: pickle.dumps(value),
        lambda value: replace(value, receipt_fingerprint="0" * 64),
    ):
        with pytest.raises(TypeError):
            operation(receipt)


@pytest.mark.parametrize(
    "locator",
    [
        "",
        "/absolute.json",
        ".",
        "..",
        "a/../geometry.json",
        "a/./geometry.json",
        "a//geometry.json",
        "geometry\x00.json",
        "geometry-\udcff.json",
    ],
)
def test_rsm_input_refuses_nonexact_locator(locator):
    with pytest.raises(TypeError, match="RSM geometry locator"):
        RSMGeometryAssetInput(locator)


def test_rsm_input_owns_mutable_pathlike_text():
    class MutablePath:
        shown = "calibration/rsm/geometry.json"

        def __fspath__(self):
            return self.shown

    mutable = MutablePath()
    request = RSMGeometryAssetInput(mutable)
    mutable.shown = "changed.json"
    assert request.locator == "calibration/rsm/geometry.json"
    assert type(request.locator) is str


@pytest.mark.parametrize("project_root", ["project\x00root", "project-\udcff-root"])
def test_rsm_capture_normalizes_hostile_project_text(project_root):
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        capture_rsm_geometry_asset(
            RSMGeometryAssetInput("geometry.json"), project_root=project_root
        )
    assert raised.value.code == "RSM_GEOMETRY_PROJECT_INVALID"


def test_rsm_capture_binds_spelling_revision_and_revalidation(tmp_path):
    project = tmp_path / "project"
    target = project / "calibration" / "rsm" / "geometry.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(canonical_rsm_geometry_resource_bytes())
    receipt = capture_rsm_geometry_asset(
        RSMGeometryAssetInput("calibration/rsm/geometry.json"),
        project_root=project,
    )
    assert receipt.lexical_relative_path == "calibration/rsm/geometry.json"
    assert receipt.resolved_relative_path == receipt.lexical_relative_path
    assert receipt.byte_count == EXPECTED_SIZE
    assert receipt.raw_sha256 == EXPECTED_SHA256
    assert receipt.semantic_fingerprint == EXPECTED_SEMANTIC
    assert len(receipt.receipt_fingerprint) == 64
    assert receipt.fingerprint == receipt.receipt_fingerprint
    assert revalidate_rsm_geometry_asset(receipt) == receipt.content

    target.touch()
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        revalidate_rsm_geometry_asset(receipt)
    assert raised.value.code == "RSM_GEOMETRY_ASSET_IDENTITY_MISMATCH"


def test_rsm_capture_refuses_project_and_asset_symlink_ancestry(tmp_path):
    raw = canonical_rsm_geometry_resource_bytes()
    real_parent = tmp_path / "real-parent"
    real_project = real_parent / "project"
    real_project.mkdir(parents=True)
    (real_project / "geometry.json").write_bytes(raw)
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        capture_rsm_geometry_asset(
            RSMGeometryAssetInput("geometry.json"),
            project_root=alias / "project",
        )
    assert raised.value.code == "RSM_GEOMETRY_SYMLINK_REFUSED"

    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "geometry.json").write_bytes(raw)
    (project / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        capture_rsm_geometry_asset(
            RSMGeometryAssetInput("linked/geometry.json"), project_root=project
        )
    assert raised.value.code == "RSM_GEOMETRY_SYMLINK_REFUSED"

    (project / "final.json").symlink_to(outside / "geometry.json")
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        capture_rsm_geometry_asset(
            RSMGeometryAssetInput("final.json"), project_root=project
        )
    assert raised.value.code == "RSM_GEOMETRY_SYMLINK_REFUSED"


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_rsm_capture_refuses_nonregular_final_node(tmp_path, kind):
    project = tmp_path / "project"
    project.mkdir()
    target = project / "geometry"
    if kind == "directory":
        target.mkdir()
    else:
        os.mkfifo(target)
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        capture_rsm_geometry_asset(
            RSMGeometryAssetInput("geometry"), project_root=project
        )
    assert raised.value.code == "RSM_GEOMETRY_NOT_REGULAR"


def test_rsm_capture_refuses_oversize_and_short_read(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    target = project / "geometry.json"
    target.write_bytes(b"x" * 65_537)
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        capture_rsm_geometry_asset(
            RSMGeometryAssetInput("geometry.json"), project_root=project
        )
    assert raised.value.code == "RSM_GEOMETRY_PARSE_FAILED"

    target.write_bytes(canonical_rsm_geometry_resource_bytes())
    real_read = os.read
    calls = 0

    def short_read(descriptor, amount):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_read(descriptor, min(16, amount))
        return b""

    monkeypatch.setattr("xrd_tools.analysis.rsm_geometry_asset.os.read", short_read)
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        capture_rsm_geometry_asset(
            RSMGeometryAssetInput("geometry.json"), project_root=project
        )
    assert raised.value.code == "RSM_GEOMETRY_ASSET_IDENTITY_MISMATCH"


def test_rsm_capture_refuses_before_after_mutation(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    target = project / "geometry.json"
    raw = canonical_rsm_geometry_resource_bytes()
    target.write_bytes(raw)
    real_states = __import__(
        "xrd_tools.analysis.rsm_geometry_asset", fromlist=["_lexical_chain_states"]
    )._lexical_chain_states

    def mutate_then_states(project_path, relative):
        target.write_bytes(raw)
        os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 1))
        return real_states(project_path, relative)

    monkeypatch.setattr(
        "xrd_tools.analysis.rsm_geometry_asset._lexical_chain_states",
        mutate_then_states,
    )
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        capture_rsm_geometry_asset(
            RSMGeometryAssetInput("geometry.json"), project_root=project
        )
    assert raised.value.code == "RSM_GEOMETRY_ASSET_IDENTITY_MISMATCH"


def test_rsm_installer_is_create_only_idempotent_and_conflict_safe(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    first = install_canonical_rsm_geometry_asset(project_root=project)
    target = project / CANONICAL_RSM_GEOMETRY_LOCATOR
    assert target.read_bytes() == canonical_rsm_geometry_resource_bytes()
    second = install_canonical_rsm_geometry_asset(project_root=project)
    assert second.receipt_fingerprint == first.receipt_fingerprint
    assert second.file_revision == first.file_revision

    conflict = tmp_path / "conflict"
    conflict_target = conflict / CANONICAL_RSM_GEOMETRY_LOCATOR
    conflict_target.parent.mkdir(parents=True)
    conflict_target.write_bytes(b"foreign")
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        install_canonical_rsm_geometry_asset(project_root=conflict)
    assert raised.value.code == "RSM_GEOMETRY_INSTALL_CONFLICT"
    assert conflict_target.read_bytes() == b"foreign"


def test_rsm_installer_refuses_symlink_parent_without_outside_write(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "calibration").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        install_canonical_rsm_geometry_asset(project_root=project)
    assert raised.value.code == "RSM_GEOMETRY_INSTALL_FAILED"
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("linked_component", ["package", "assets", "rsm", "final"])
def test_canonical_rsm_resource_refuses_symlink_ancestry(
    tmp_path, monkeypatch, linked_component
):
    raw = canonical_rsm_geometry_resource_bytes()
    real_package = tmp_path / "real-package"
    resource = real_package.joinpath(*("assets", "rsm", _RESOURCE_NAME))
    resource.parent.mkdir(parents=True)
    resource.write_bytes(raw)
    package = tmp_path / "package"
    if linked_component == "package":
        package.symlink_to(real_package, target_is_directory=True)
    else:
        package.mkdir()
        if linked_component == "assets":
            (package / "assets").symlink_to(
                real_package / "assets", target_is_directory=True
            )
        elif linked_component == "rsm":
            (package / "assets").mkdir()
            (package / "assets" / "rsm").symlink_to(
                real_package / "assets" / "rsm", target_is_directory=True
            )
        else:
            target = package.joinpath("assets", "rsm", _RESOURCE_NAME)
            target.parent.mkdir(parents=True)
            target.symlink_to(resource)
    monkeypatch.setattr(
        "xrd_tools.analysis.rsm_geometry_asset.resources.files",
        lambda _package: package,
    )
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        canonical_rsm_geometry_resource_bytes()
    assert raised.value.code == "RSM_CANONICAL_ASSET_UNAVAILABLE"


_RESOURCE_NAME = "ssrl17_2_psic_pilatus300k_sto_v1.json"


@pytest.mark.parametrize("package_node", [object(), "package\x00root"])
def test_canonical_rsm_resource_refuses_nonfilesystem_root(monkeypatch, package_node):
    monkeypatch.setattr(
        "xrd_tools.analysis.rsm_geometry_asset.resources.files",
        lambda _package: package_node,
    )
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        canonical_rsm_geometry_resource_bytes()
    assert raised.value.code == "RSM_CANONICAL_ASSET_UNAVAILABLE"


def test_canonical_rsm_resource_refuses_dotdot_package_spelling(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    hostile = os.path.join(os.fspath(package), os.pardir, package.name)
    monkeypatch.setattr(
        "xrd_tools.analysis.rsm_geometry_asset.resources.files",
        lambda _package: hostile,
    )
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        canonical_rsm_geometry_resource_bytes()
    assert raised.value.code == "RSM_CANONICAL_ASSET_UNAVAILABLE"


def test_rsm_parser_is_engine_light(monkeypatch):
    imported = []
    real_import = __import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"numpy", "pyFAI", "xrayutilities"}:
            imported.append(name)
            raise AssertionError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    projection = parse_rsm_geometry_asset_bytes(canonical_rsm_geometry_resource_bytes())
    assert projection.semantic_fingerprint == EXPECTED_SEMANTIC
    assert imported == []
