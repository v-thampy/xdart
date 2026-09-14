"""xrd_tools.io.descriptor_path: the no-dir_fd installers' placement check
and handle-held disposal (Codex PR #1 review F2; Codex review of 0ed7a46c F2),
and the capture-chain drift report (PR #1 round 12, macos-15-intel)."""

from __future__ import annotations

import os
import sys

import pytest

from xrd_tools.io import descriptor_path


def _symlink(link, target):
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symbolic links unavailable here: {error}")


def test_exclusive_create_refuses_an_existing_entry_and_a_planted_link(tmp_path):
    leaf = tmp_path / "leaf.json"
    descriptor = descriptor_path.create_exclusive_leaf(os.fspath(leaf))
    try:
        assert os.write(descriptor, b"payload") == 7
    finally:
        os.close(descriptor)
    assert leaf.read_bytes() == b"payload"
    with pytest.raises(FileExistsError):
        descriptor_path.create_exclusive_leaf(os.fspath(leaf))
    assert leaf.read_bytes() == b"payload"

    link = tmp_path / "link.json"
    dangling = tmp_path / "never-created.json"
    _symlink(link, dangling)
    with pytest.raises(FileExistsError):
        descriptor_path.create_exclusive_leaf(os.fspath(link))
    assert link.is_symlink()
    assert not dangling.exists()


def test_misplacement_holds_the_landing_directory_and_the_entry_name(tmp_path):
    inspected = tmp_path / "inspected"
    other = tmp_path / "other"
    inspected.mkdir()
    other.mkdir()
    identity = descriptor_path.directory_identity(os.lstat(inspected))
    descriptor = descriptor_path.create_exclusive_leaf(
        os.fspath(inspected / "leaf.json")
    )
    try:
        assert descriptor_path.created_leaf_misplacement(
            descriptor, identity, "leaf.json"
        ) is None
        wrong_name = descriptor_path.created_leaf_misplacement(
            descriptor, identity, "other.json"
        )
        assert wrong_name is not None and wrong_name.endswith(
            ": not the requested entry"
        )
        wrong_directory = descriptor_path.created_leaf_misplacement(
            descriptor,
            descriptor_path.directory_identity(os.lstat(other)),
            "leaf.json",
        )
        assert wrong_directory is not None and wrong_directory.endswith(
            ": outside the inspected directory"
        )
        # Renaming the inspected directory keeps its identity: still inside.
        renamed = tmp_path / "renamed"
        inspected.rename(renamed)
        assert descriptor_path.created_leaf_misplacement(
            descriptor, identity, "leaf.json"
        ) is None
    finally:
        os.close(descriptor)


def test_disposal_goes_through_the_handle_or_reports_it_cannot(tmp_path):
    leaf = tmp_path / "leaf.json"
    descriptor = descriptor_path.create_exclusive_leaf(os.fspath(leaf))
    try:
        disposed = descriptor_path.dispose_created_leaf(descriptor)
    finally:
        os.close(descriptor)
    if sys.platform == "win32":
        assert disposed
        assert not leaf.exists()
    else:
        assert not disposed
        assert leaf.stat().st_size == 0


# (mode, dev, ino, size, mtime_ns, ctime_ns), as the capture walkers record.
_FIELDS = ("mode", "dev", "ino", "size", "mtime_ns", "ctime_ns")
_ROOT = (0o40755, 7, 2, 640, 1_000, 1_000)
_DIRECTORY = (0o40755, 7, 3, 96, 2_000, 2_000)
_LEAF = (0o100644, 7, 4, 4_837, 3_000, 3_000)
_COMPONENTS = ("/", "/project", "/project/surface.json")


def _with(state, **fields):
    return tuple(fields.get(name, value) for name, value in zip(_FIELDS, state))


def test_chain_drift_holds_ancestors_to_identity_and_the_leaf_to_its_state():
    opened = (_ROOT, _DIRECTORY, _LEAF)
    assert descriptor_path.chain_drift(opened, opened, _COMPONENTS) is None
    # A sibling entry created or removed beside the chain: the ancestors'
    # size, mtime and ctime move while their identity does not.
    beside = (
        _with(_ROOT, size=672, mtime_ns=9_000, ctime_ns=9_000),
        _with(_DIRECTORY, size=128, mtime_ns=9_500, ctime_ns=9_500),
        _LEAF,
    )
    assert descriptor_path.chain_drift(opened, beside, _COMPONENTS) is None
    # The leaf keeps every recorded slot.
    for field, value in (
        ("size", 4_838), ("mtime_ns", 3_001), ("ctime_ns", 3_001), ("mode", 0o100600),
    ):
        moved = (_ROOT, _DIRECTORY, _with(_LEAF, **{field: value}))
        assert descriptor_path.chain_drift(opened, moved, _COMPONENTS) == (
            f"leaf /project/surface.json: {field} "
            f"{_LEAF[_FIELDS.index(field)]} -> {value}"
        )


def test_chain_drift_names_the_first_ancestor_whose_identity_moved():
    opened = (_ROOT, _DIRECTORY, _LEAF)
    for field, value in (("ino", 30), ("dev", 8), ("mode", 0o40700)):
        swapped = (_ROOT, _with(_DIRECTORY, **{field: value}), _with(_LEAF, ino=40))
        assert descriptor_path.chain_drift(opened, swapped, _COMPONENTS) == (
            f"ancestor /project: {field} "
            f"{_DIRECTORY[_FIELDS.index(field)]} -> {value}"
        )
    # A chain that gained or lost a slot cannot be the same chain, and a
    # caller that names the wrong number of components is refused, not
    # trusted.
    assert descriptor_path.chain_drift(
        opened, opened[:2], _COMPONENTS
    ) == "chain length 3 -> 2"
    assert descriptor_path.chain_drift(
        opened, opened, _COMPONENTS[:2]
    ) == "chain of 3 slots names 2 components"


def test_chain_components_name_every_slot_the_walkers_record():
    project = os.path.join(os.sep, "srv", "project")
    assert descriptor_path.chain_components(
        project, os.path.join("calibration", "xu", "surface.json")
    ) == (
        os.sep,
        os.path.join(os.sep, "srv"),
        project,
        os.path.join(project, "calibration"),
        os.path.join(project, "calibration", "xu"),
        os.path.join(project, "calibration", "xu", "surface.json"),
    )
