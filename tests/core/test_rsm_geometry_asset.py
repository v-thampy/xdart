from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import stat
import subprocess
import sys

import pytest

from xrd_tools.analysis.canonical_fingerprint import (
    analysis_canonical_fingerprint,
)
import xrd_tools.analysis.rsm_geometry_asset as rsm_asset_module
from xrd_tools.analysis.rsm_geometry_asset import (
    CANONICAL_RSM_GEOMETRY_LOCATOR,
    RSMEffectiveGeometry,
    RSMGeometryAssetInput,
    RSMGeometryAssetRefused,
    RSMMemberGeometryBinding,
    bind_rsm_member_geometry,
    canonical_rsm_geometry_resource_bytes,
    capture_rsm_geometry_asset,
    install_canonical_rsm_geometry_asset,
    lower_rsm_effective_geometry,
    parse_rsm_geometry_asset_bytes,
    revalidate_rsm_geometry_asset,
    rsm_effective_pixel_q_map,
    rsm_geometry_asset_input,
)


EXPECTED_SIZE = 745
EXPECTED_SHA256 = (
    "0f22b00363ff93fec7b97c7b5c31f8b8e389aac3aae334cbadbc213b629d9f84"
)
EXPECTED_SEMANTIC = (
    "a5841542c58c5ff33899ef13a6f3883c6f6e356eefb11c41d1f24936eed9f0fa"
)
FROZEN_VIEW_DIAGNOSTIC = (
    "036cff52be68a8c4aa3eb6e96be4b02d56e6f230a9e076353fabe96011927b5c"
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
    assert analysis_canonical_fingerprint(
        "rsm-geometry-asset-v1", json.loads(raw)
    ) == EXPECTED_SEMANTIC
    assert analysis_canonical_fingerprint(
        "rsm-geometry-asset-v1", projection.value
    ) == FROZEN_VIEW_DIAGNOSTIC
    assert FROZEN_VIEW_DIAGNOSTIC != projection.semantic_fingerprint

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
        rsm_geometry_asset_input("geometry.json"), project_root=project
    )
    for operation in (
        copy.copy,
        copy.deepcopy,
        lambda value: pickle.dumps(value),
        lambda value: replace(value, receipt_fingerprint="0" * 64),
    ):
        with pytest.raises(TypeError):
            operation(receipt)


def test_rsm_input_is_factory_owned_noncopyable_and_nonreplaceable():
    with pytest.raises(TypeError, match="factory-owned"):
        RSMGeometryAssetInput("geometry.json")

    request = rsm_geometry_asset_input("geometry.json")
    for operation in (
        copy.copy,
        copy.deepcopy,
        lambda value: pickle.dumps(value),
        lambda value: replace(value),
        lambda value: replace(value, locator="other.json"),
        lambda value: copy.replace(value),
        lambda value: copy.replace(value, locator="other.json"),
    ):
        with pytest.raises(TypeError):
            operation(request)


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
        "x" * 10_000,
    ],
)
def test_rsm_input_refuses_nonexact_locator(locator):
    with pytest.raises(TypeError, match="RSM geometry locator"):
        rsm_geometry_asset_input(locator)


def test_rsm_input_owns_mutable_pathlike_text():
    class MutablePath:
        shown = "calibration/rsm/geometry.json"

        def __fspath__(self):
            return self.shown

    mutable = MutablePath()
    request = rsm_geometry_asset_input(mutable)
    mutable.shown = "changed.json"
    assert request.locator == "calibration/rsm/geometry.json"
    assert type(request.locator) is str


def test_rsm_input_enforces_exact_utf8_byte_bound():
    exact = "a/" + "é" * 2_047
    assert len(exact.encode("utf-8")) == 4_096
    assert rsm_geometry_asset_input(exact).locator == exact
    with pytest.raises(TypeError, match="4096 UTF-8 bytes"):
        rsm_geometry_asset_input(exact + "x")


@pytest.mark.parametrize(
    "project_root",
    ["project\x00root", "project-\udcff-root", "x" * 10_000],
)
def test_rsm_capture_normalizes_hostile_project_text(project_root):
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        capture_rsm_geometry_asset(
            rsm_geometry_asset_input("geometry.json"), project_root=project_root
        )
    assert raised.value.code == "RSM_GEOMETRY_PROJECT_INVALID"


def test_rsm_capture_binds_spelling_revision_and_revalidation(tmp_path):
    project = tmp_path / "project"
    target = project / "calibration" / "rsm" / "geometry.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(canonical_rsm_geometry_resource_bytes())
    receipt = capture_rsm_geometry_asset(
        rsm_geometry_asset_input("calibration/rsm/geometry.json"),
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
            rsm_geometry_asset_input("geometry.json"),
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
            rsm_geometry_asset_input("linked/geometry.json"), project_root=project
        )
    assert raised.value.code == "RSM_GEOMETRY_SYMLINK_REFUSED"

    (project / "final.json").symlink_to(outside / "geometry.json")
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        capture_rsm_geometry_asset(
            rsm_geometry_asset_input("final.json"), project_root=project
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
            rsm_geometry_asset_input("geometry"), project_root=project
        )
    assert raised.value.code == "RSM_GEOMETRY_NOT_REGULAR"


def test_rsm_capture_refuses_oversize_and_short_read(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    target = project / "geometry.json"
    target.write_bytes(b"x" * 65_537)
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        capture_rsm_geometry_asset(
            rsm_geometry_asset_input("geometry.json"), project_root=project
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
            rsm_geometry_asset_input("geometry.json"), project_root=project
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
            rsm_geometry_asset_input("geometry.json"), project_root=project
        )
    assert raised.value.code == "RSM_GEOMETRY_ASSET_IDENTITY_MISMATCH"


def test_rsm_capture_refuses_exact_byte_inode_swap_before_open(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    project.mkdir()
    target = project / "geometry.json"
    raw = canonical_rsm_geometry_resource_bytes()
    target.write_bytes(raw)
    real_open = rsm_asset_module._open_no_follow_chain
    held = []

    def swap_then_open(project_path, relative, **kwargs):
        held.append(target.open("rb"))
        target.unlink()
        target.write_bytes(raw)
        return real_open(project_path, relative, **kwargs)

    monkeypatch.setattr(rsm_asset_module, "_open_no_follow_chain", swap_then_open)
    try:
        with pytest.raises(RSMGeometryAssetRefused) as raised:
            capture_rsm_geometry_asset(
                rsm_geometry_asset_input("geometry.json"), project_root=project
            )
        assert raised.value.code == "RSM_GEOMETRY_ASSET_IDENTITY_MISMATCH"
    finally:
        for stream in held:
            stream.close()


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


def test_rsm_installer_never_unlinks_foreign_name_race_replacement(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    project.mkdir()
    target = project / CANONICAL_RSM_GEOMETRY_LOCATOR
    foreign = b"foreign replacement"
    attacked = False

    def replace_name_then_fail(_descriptor, _payload):
        nonlocal attacked
        assert not attacked
        attacked = True
        target.unlink()
        target.write_bytes(foreign)
        raise OSError("synthetic write failure")

    monkeypatch.setattr(rsm_asset_module.os, "write", replace_name_then_fail)
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        install_canonical_rsm_geometry_asset(project_root=project)
    assert attacked
    assert raised.value.code == "RSM_GEOMETRY_INSTALL_FAILED"
    assert target.read_bytes() == foreign


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


def test_canonical_rsm_resource_refuses_exact_byte_inode_swap_before_open(
    tmp_path, monkeypatch
):
    raw = canonical_rsm_geometry_resource_bytes()
    package = tmp_path / "package"
    target = package / "assets" / "rsm" / _RESOURCE_NAME
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)
    real_open = rsm_asset_module._open_no_follow_chain
    held = []

    def swap_then_open(project_path, relative, **kwargs):
        held.append(target.open("rb"))
        target.unlink()
        target.write_bytes(raw)
        return real_open(project_path, relative, **kwargs)

    monkeypatch.setattr(rsm_asset_module.resources, "files", lambda _name: package)
    monkeypatch.setattr(rsm_asset_module, "_open_no_follow_chain", swap_then_open)
    try:
        with pytest.raises(RSMGeometryAssetRefused) as raised:
            canonical_rsm_geometry_resource_bytes()
        assert raised.value.code == "RSM_CANONICAL_ASSET_UNAVAILABLE"
    finally:
        for stream in held:
            stream.close()


def test_extracted_canonical_fingerprint_preserves_golden_identities():
    import numpy as np
    from xrd_tools.analysis import canonical_fingerprint as helper
    from xrd_tools.analysis import scan_operations

    assert scan_operations.analysis_canonical_fingerprint is (
        helper.analysis_canonical_fingerprint
    )
    assert scan_operations._digest is helper._digest
    assert scan_operations._canonical_charge is helper._canonical_charge
    assert helper._PublicFingerprintContainer.__module__ == (
        "xrd_tools.analysis.scan_operations"
    )
    values = {
        "mapping": analysis_canonical_fingerprint("x", {"b": 2, "a": 1}),
        "list": analysis_canonical_fingerprint("x", [1, 2]),
        "tuple": analysis_canonical_fingerprint("x", (1, 2)),
        "path": analysis_canonical_fingerprint("x", Path("a")),
        "array": analysis_canonical_fingerprint(
            "x", np.array([[1, 2], [3, 4]], dtype="<i4")
        ),
        "missing": analysis_canonical_fingerprint(
            "x", np.array([float("nan")], dtype="<f8"), allow_missing=True
        ),
    }
    assert values == {
        "mapping": "dda8fa340049c735335394b4f838f45ce43dcd4f7651b05a757532ac15c2ecb4",
        "list": "a186a2ee3e183472b4ae1b11a0015253535a0c372b94313fd4eccd701d025828",
        "tuple": "3a196003d76aae95e8a175e33dbfc7ae685c7f4138b47363418da942b607a83e",
        "path": "a7c34e84160463ddb5ee58213f7ea4a932523b2c73329e3d6e786c06af69f806",
        "array": "79cddde133016cac593096fa33aa6850ee40521e5a8818fd77f524f27279b6b0",
        "missing": "be58818851b8c9b474b8ce2e2fd27f0764b7fdd77ae5cddc2797e6dc1e955e23",
    }


def test_rsm_parser_and_public_fingerprint_are_fresh_import_engine_light():
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    source_root = os.fspath(repository / "src")
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not inherited else source_root + os.pathsep + inherited
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    program = f"""
import sys
from xrd_tools.analysis.rsm_geometry_asset import (
    canonical_rsm_geometry_resource_bytes,
    parse_rsm_geometry_asset_bytes,
)
projection = parse_rsm_geometry_asset_bytes(canonical_rsm_geometry_resource_bytes())
assert projection.semantic_fingerprint == {EXPECTED_SEMANTIC!r}
from xrd_tools.analysis import analysis_canonical_fingerprint
assert analysis_canonical_fingerprint('x', {{'b': 2, 'a': 1}}) == \
    'dda8fa340049c735335394b4f838f45ce43dcd4f7651b05a757532ac15c2ecb4'
forbidden = sorted(
    name for name in sys.modules
    if name.split('.', 1)[0] in {{'numpy', 'pyFAI', 'xrayutilities'}}
)
assert forbidden == [], forbidden
assert 'xrd_tools.analysis.scan_operations' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _installed_effective_geometry(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    receipt = install_canonical_rsm_geometry_asset(project_root=project)
    return receipt, lower_rsm_effective_geometry(receipt)


def test_rsm_effective_geometry_lowers_every_exact_psic_and_runtime_fact(tmp_path):
    from types import MappingProxyType

    from xrd_tools.core.geometry import (
        DetectorHeader,
        Diffractometer,
        ImageOrientation,
        PixelQMap,
    )
    from xrd_tools.core.geometry.xu_runtime import (
        XuRuntimeRequirements,
        xu_runtime_requirements_projection,
    )

    receipt, effective = _installed_effective_geometry(tmp_path)
    assert type(effective) is RSMEffectiveGeometry
    assert type(effective.diffractometer_projection) is Diffractometer
    assert effective.diffractometer_projection == Diffractometer.psic()
    assert all(
        isinstance(
            getattr(effective.diffractometer_projection, name),
            MappingProxyType,
        )
        for name in ("qconv_kwargs", "hxrd_kwargs", "ang2q_kwargs")
    )
    assert effective.detector_header == DetectorHeader(
        cch1=97.0,
        cch2=243.0,
        pwidth1=0.172,
        pwidth2=0.172,
        distance=1014.7173,
        Nch1=195,
        Nch2=487,
    )
    assert effective.image_orientation == ImageOrientation()
    assert effective.roi == (0, -1, 0, -1)
    assert effective.runtime_requirements == XuRuntimeRequirements()
    assert xu_runtime_requirements_projection(effective.runtime_requirements) == (
        "1.7.12",
        "1.7.12",
        "2.5.1",
        1e-8,
        8,
        "CPython",
        "3.13.14",
        "Darwin",
        "arm64",
        "shared_xrd_tools_xu_rlock_v1",
        1,
    )
    psic_projection = (
        "psic",
        (
            ("nu", 1.0, 0.0),
            ("del", 1.0, 0.0),
            ("", 1.0, 0.0),
            ("eta", 1.0, 0.0),
        ),
        ("x+", "z-", "y+", "z-"),
        ("x+", "z-"),
        (0.0, 1.0, 0.0),
        ("z-", "x-"),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        "real",
        tuple((role, 1.0, 0.0) for role in ("mu", "eta", "chi", "phi", "nu", "del")),
        ("eta", "chi", "phi", "mu"),
        ("del", "nu"),
        (),
        (),
        (),
        None,
    )
    expected = analysis_canonical_fingerprint(
        "rsm-effective-geometry-v1",
        (
            receipt.receipt_fingerprint,
            EXPECTED_SEMANTIC,
            psic_projection,
            (97.0, 243.0, 0.172, 0.172, 1014.7173, 195, 487),
            (0, False, False, False),
            (0, -1, 0, -1),
            xu_runtime_requirements_projection(XuRuntimeRequirements()),
        ),
    )
    assert effective.fingerprint == expected
    mapper = rsm_effective_pixel_q_map(effective)
    assert type(mapper) is PixelQMap
    assert mapper.diff_config is effective.diffractometer_projection
    assert mapper.header is effective.detector_header


def test_rsm_effective_geometry_is_owned_nested_immutable_and_revalidated(tmp_path):
    receipt, effective = _installed_effective_geometry(tmp_path)
    selectors = tuple(
        (role, __import__(
            "xrd_tools.analysis.module_transaction",
            fromlist=["MetadataColumnSelector"],
        ).MetadataColumnSelector(role, 0))
        for role in ("mu", "eta", "chi", "phi", "nu", "del")
    )
    binding = bind_rsm_member_geometry(
        effective, member_ordinal=0, motor_selectors=selectors
    )
    with pytest.raises(TypeError, match="factory-owned"):
        RSMEffectiveGeometry(
            effective.asset_receipt_fingerprint,
            effective.asset_semantic_fingerprint,
            effective.diffractometer_projection,
            effective.detector_header,
            effective.image_orientation,
            effective.roi,
            effective.runtime_requirements,
            effective.fingerprint,
        )
    with pytest.raises(TypeError, match="factory-owned"):
        RSMMemberGeometryBinding(
            binding.member_ordinal,
            binding.effective_geometry_fingerprint,
            binding.motor_selectors,
            binding.fingerprint,
        )
    for value in (effective, binding):
        for operation in (
            copy.copy,
            copy.deepcopy,
            lambda item: pickle.dumps(item),
            lambda item: replace(item),
            lambda item: copy.replace(item),
        ):
            with pytest.raises(TypeError):
                operation(value)
    for name in ("qconv_kwargs", "hxrd_kwargs", "ang2q_kwargs"):
        with pytest.raises(TypeError):
            getattr(effective.diffractometer_projection, name)["foreign"] = 1

    target = Path(receipt.project_root) / receipt.lexical_relative_path
    target.touch()
    with pytest.raises(RSMGeometryAssetRefused) as raised:
        lower_rsm_effective_geometry(receipt)
    assert raised.value.code == "RSM_GEOMETRY_ASSET_IDENTITY_MISMATCH"


def test_rsm_effective_geometry_binds_receipt_custody_separately_from_semantics(
    tmp_path,
):
    first_project = tmp_path / "first"
    second_project = tmp_path / "second"
    first_project.mkdir()
    second_project.mkdir()
    first_receipt = install_canonical_rsm_geometry_asset(
        project_root=first_project
    )
    second_receipt = install_canonical_rsm_geometry_asset(
        project_root=second_project
    )
    assert first_receipt.semantic_fingerprint == second_receipt.semantic_fingerprint
    assert first_receipt.receipt_fingerprint != second_receipt.receipt_fingerprint
    first = lower_rsm_effective_geometry(first_receipt)
    second = lower_rsm_effective_geometry(second_receipt)
    assert first.asset_semantic_fingerprint == second.asset_semantic_fingerprint
    assert first.asset_receipt_fingerprint != second.asset_receipt_fingerprint
    assert first.fingerprint != second.fingerprint


def test_rsm_effective_lowering_refuses_drift_in_every_psic_projection_group(
    tmp_path, monkeypatch
):
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.core.geometry.diffractometer import AngleMapping

    project = tmp_path / "project"
    project.mkdir()
    receipt = install_canonical_rsm_geometry_asset(project_root=project)
    base = Diffractometer.psic()
    variants = (
        replace(base, preset="foreign"),
        replace(base, rot1=replace(base.rot1, offset=1.0)),
        replace(base, rot2=replace(base.rot2, source_motor="foreign")),
        replace(base, rot3=AngleMapping(source_motor="foreign")),
        replace(base, incident_angle=replace(base.incident_angle, sign=-1.0)),
        replace(base, sample_circles=tuple(reversed(base.sample_circles))),
        replace(base, detector_circles=tuple(reversed(base.detector_circles))),
        replace(base, r_i=(1.0, 0.0, 0.0)),
        replace(base, camera=("x+", "z-")),
        replace(base, hxrd_n=(1.0, 0.0, 0.0)),
        replace(base, hxrd_q=(1.0, 0.0, 0.0)),
        replace(base, hxrd_geometry="reciprocal"),
        replace(
            base,
            circle_motors=(
                replace(base.circle_motors[0], offset=1.0),
                *base.circle_motors[1:],
            ),
        ),
        replace(base, sample_motors=tuple(reversed(base.sample_motors))),
        replace(base, detector_motors=tuple(reversed(base.detector_motors))),
        replace(base, qconv_kwargs={"foreign": 1}),
        replace(base, hxrd_kwargs={"foreign": 1}),
        replace(base, ang2q_kwargs={"foreign": 1}),
        replace(base, calibration=object()),
    )
    for variant in variants:
        with monkeypatch.context() as isolated:
            isolated.setattr(
                Diffractometer,
                "psic",
                classmethod(lambda _cls, _variant=variant, **_kwargs: _variant),
            )
            with pytest.raises(RSMGeometryAssetRefused) as raised:
                lower_rsm_effective_geometry(receipt)
            assert raised.value.code == "RSM_GEOMETRY_LOWERING_MISMATCH"


def test_rsm_member_geometry_binding_is_ordered_occurrence_sensitive_and_local(
    tmp_path,
):
    from xrd_tools.analysis.module_transaction import MetadataColumnSelector

    _receipt, effective = _installed_effective_geometry(tmp_path)
    roles = ("mu", "eta", "chi", "phi", "nu", "del")

    def selectors(*, eta_occurrence=0):
        return tuple(
            (
                role,
                MetadataColumnSelector(
                    role, eta_occurrence if role == "eta" else 0
                ),
            )
            for role in roles
        )

    member0 = bind_rsm_member_geometry(
        effective, member_ordinal=0, motor_selectors=selectors()
    )
    member1 = bind_rsm_member_geometry(
        effective, member_ordinal=1, motor_selectors=selectors()
    )
    member1_eta1 = bind_rsm_member_geometry(
        effective, member_ordinal=1, motor_selectors=selectors(eta_occurrence=1)
    )
    repeated0 = bind_rsm_member_geometry(
        effective, member_ordinal=0, motor_selectors=selectors()
    )
    assert type(member0) is RSMMemberGeometryBinding
    assert member0.fingerprint == repeated0.fingerprint
    assert member0.fingerprint != member1.fingerprint
    assert member1.fingerprint != member1_eta1.fingerprint
    assert member0.fingerprint == repeated0.fingerprint
    assert MetadataColumnSelector("Seconds", 7) not in tuple(
        selector for _role, selector in member0.motor_selectors
    )

    with pytest.raises(TypeError):
        bind_rsm_member_geometry(
            effective, member_ordinal=True, motor_selectors=selectors()
        )
    with pytest.raises(TypeError):
        bind_rsm_member_geometry(
            effective,
            member_ordinal=16,
            motor_selectors=selectors(),
        )
    with pytest.raises(TypeError):
        bind_rsm_member_geometry(
            effective,
            member_ordinal=0,
            motor_selectors=(selectors()[1], selectors()[0], *selectors()[2:]),
        )
    duplicate = list(selectors())
    duplicate[1] = ("eta", duplicate[0][1])
    with pytest.raises(ValueError):
        bind_rsm_member_geometry(
            effective,
            member_ordinal=0,
            motor_selectors=tuple(duplicate),
        )


def test_rsm_effective_mapper_is_exactly_q_equivalent_to_legacy_psic(tmp_path):
    import numpy as np

    from xrd_tools.core.geometry import Diffractometer, PixelQMap

    _receipt, effective = _installed_effective_geometry(tmp_path)
    effective_mapper = rsm_effective_pixel_q_map(effective)
    legacy_mapper = PixelQMap(Diffractometer.psic(), effective.detector_header)
    angles = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            (0.0, 0.1),
            (10.0, 10.1),
            (90.0, 90.0),
            (0.0, 0.0),
            (1.0, 1.1),
            (20.0, 20.1),
        )
    )
    effective_q = effective_mapper.pixel_q(
        angles,
        13_000.007,
        UB=np.eye(3),
        roi=effective.roi,
    )
    legacy_q = legacy_mapper.pixel_q(
        angles,
        13_000.007,
        UB=np.eye(3),
        roi=effective.roi,
    )
    assert all(
        np.array_equal(effective_axis, legacy_axis)
        for effective_axis, legacy_axis in zip(effective_q, legacy_q, strict=True)
    )
