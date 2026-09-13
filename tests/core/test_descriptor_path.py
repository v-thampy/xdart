"""xrd_tools.io.descriptor_path: the no-dir_fd installers' placement check
and handle-held disposal (Codex PR #1 review F2; Codex review of 0ed7a46c F2)."""

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
