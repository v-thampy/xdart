"""The raw-file guard's folder-by-folder keys equal per-file canonical paths (NEXT-1215-SHARE-LOOKUPS).

``output_preflight._resolved_raw_keys`` resolves each folder once and builds a plain file's key from its folder's
canonical path and its listed name; links, reparse points, unlisted files and unreadable folders are resolved one by
one. On every platform its keys must equal ``normcase(realpath(path))`` for each path, which the guard compared before
(maintainer, 2026-11-19: his pick "Cheaper raw-file guard"). Runs natively in the Windows CI job, where the listing's
reparse-point attribute is read."""
import os
import sys

import numpy as np
import pytest

from xdart.gui.tabs.scattering import output_preflight


def _expected(paths):
    return {os.path.normcase(os.path.realpath(path)) for path in paths}


def _write(path):
    path.write_bytes(np.arange(16, dtype=np.uint16).tobytes())
    return path


def test_plain_files_mixed_case_names_and_missing_files(tmp_path):
    root = tmp_path / "Samples"
    (root / "ScanA").mkdir(parents=True)
    paths = [_write(root / "a_0001.tif"), _write(root / "B_0002.TIF"), _write(root / "ScanA" / "c_0001.tif"),
             root / "missing_0001.tif"]
    assert output_preflight._resolved_raw_keys(paths) == _expected(paths)


def test_a_file_link_is_resolved_on_its_own(tmp_path):
    target = _write(tmp_path / "elsewhere.tif")
    folder = tmp_path / "raw"
    folder.mkdir()
    link = folder / "linked_0001.tif"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"file symlinks unavailable: {error}")
    plain = _write(folder / "plain_0001.tif")
    keys = output_preflight._resolved_raw_keys([link, plain])
    assert keys == _expected([link, plain])
    assert os.path.normcase(os.path.realpath(target)) in keys


def test_a_folded_name_cannot_borrow_a_plain_entry_for_a_link(tmp_path, monkeypatch):
    plain = _write(tmp_path / "plain.tif")
    referent = _write(tmp_path / "referent.tif")
    link = tmp_path / "link.tif"
    try:
        link.symlink_to(referent)
    except OSError as error:
        pytest.skip(f"file symlinks unavailable: {error}")
    normcase = os.path.normcase

    def folded(path):
        # Model two case-distinct directory entries with the same Windows fold.
        if path in (plain.name, link.name):
            return plain.name
        return normcase(path)

    monkeypatch.setattr(os.path, "normcase", folded)
    assert output_preflight._resolved_raw_keys([link]) == _expected([link])


def test_an_unresolved_parent_falls_back_to_the_file(tmp_path, monkeypatch):
    folder = tmp_path / "samples"
    folder.mkdir()
    path = _write(folder / "image.tif")
    realpath = os.path.realpath
    expected = os.path.normcase(realpath(path))

    def resolve(value, *, strict=False):
        # Simulate a provider that cannot resolve the folder, but can resolve its files.
        if os.fspath(value) == os.fspath(folder):
            if strict:
                raise PermissionError(os.fspath(value))
            return str(tmp_path / "unresolved")
        return realpath(value, strict=strict)

    monkeypatch.setattr(os.path, "realpath", resolve)
    assert output_preflight._resolved_raw_keys([path]) == {expected}


def test_a_long_windows_leaf_can_keep_a_prefix_its_parent_dropped(tmp_path, monkeypatch):
    folder = tmp_path / "mapped"
    folder.mkdir()
    path = _write(folder / "image.tif")
    expanded = str(tmp_path / ("a" * 100) / ("\U00010400" * 90))
    leaf = "\\\\?\\" + os.path.join(expanded, path.name)
    realpath = os.path.realpath
    scandir = os.scandir

    def resolve(value, *, strict=False):
        if os.fspath(value) == str(folder):
            return expanded
        if os.fspath(value) == str(path):
            return leaf
        return realpath(value, strict=strict)

    def listing(value):
        return scandir(folder if os.fspath(value) == expanded else value)

    monkeypatch.setattr(output_preflight, "_CHECK_RESOLVED_RAW_PATHS", True)
    monkeypatch.setattr(os.path, "realpath", resolve)
    monkeypatch.setattr(os, "scandir", listing)
    assert output_preflight._resolved_raw_keys([path]) == _expected([path])


def test_a_folder_reached_through_a_link_or_junction_resolves_to_its_target(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    _write(real / "a_0001.tif")
    alias = tmp_path / "alias"
    try:
        if sys.platform == "win32":
            import _winapi
            _winapi.CreateJunction(str(real), str(alias))
        else:
            alias.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError) as error:
        pytest.skip(f"folder links unavailable: {error}")
    paths = [alias / "a_0001.tif"]
    assert output_preflight._resolved_raw_keys(paths) == _expected(paths) == _expected([real / "a_0001.tif"])


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs a POSIX folder this user cannot list")
def test_a_folder_that_cannot_be_listed_falls_back_to_per_file_resolution(tmp_path):
    folder = tmp_path / "locked"
    folder.mkdir()
    path = _write(folder / "a_0001.tif")
    folder.chmod(0o300)  # enterable, not listable
    try:
        assert output_preflight._resolved_raw_keys([path]) == _expected([path])
    finally:
        folder.chmod(0o700)
