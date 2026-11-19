"""The Folder listing gives the same candidates as before it stopped asking about every file (NEXT-1215-SHARE-LOOKUPS).

``_directory_matching_paths`` used ``rglob`` plus ``Path.is_file()`` for every entry, one stat each, sidecars included;
it now filters names first and asks the listing's own entries. The reference below is the 1.2.14 algorithm, run over
the same tree in both modes."""
import os
from pathlib import Path

import pytest

from xdart.gui.tabs.scattering import output_preflight
from xrd_tools.sources.selection import DirectorySourceSpec


def _reference(source, skip=()):
    """1.2.14's ``_directory_matching_paths``, verbatim but for the cancel check."""
    root = Path(source.root).expanduser().absolute()
    suffixes = tuple(str(value).casefold() for value in source.suffixes)
    name_ok = output_preflight.compile_filter(source.name_filter)
    iterator = root.rglob("*") if source.recursive else root.iterdir()
    paths = []
    for path in iterator:
        if not path.is_file():
            continue
        try:
            if len(path.absolute().relative_to(root).parts) > 2:
                continue
        except ValueError:
            continue
        if skip and output_preflight._under_skipped(path, skip):
            continue
        low = path.name.casefold()
        suffix = next((value for value in suffixes if low.endswith(value)), "")
        if suffixes and not suffix:
            continue
        stem = path.name[:-len(suffix)] if suffix else path.name
        if name_ok(stem):
            paths.append(path.absolute())
    return tuple(sorted(dict.fromkeys(paths)))


def _tree(tmp_path):
    root = tmp_path / "samples"
    (root / "scanA").mkdir(parents=True)
    (root / "scanA" / "deeper").mkdir()
    (root / "processed").mkdir()
    (root / "a.tif").write_bytes(b"x")
    (root / "a.txt").write_text("x")
    (root / "B.TIF").write_bytes(b"x")
    (root / ".hidden.tif").write_bytes(b"x")
    (root / "folder.tif").mkdir()
    (root / "scanA" / "c.tif").write_bytes(b"x")
    (root / "scanA" / "c.txt").write_text("x")
    (root / "scanA" / "deeper" / "d.tif").write_bytes(b"x")
    (root / "processed" / "e.tif").write_bytes(b"x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "f.tif").write_bytes(b"x")
    links = True
    try:
        (root / "linked_folder").symlink_to(outside, target_is_directory=True)
        (root / "linked_file.tif").symlink_to(outside / "f.tif")
        (root / "dangling.tif").symlink_to(tmp_path / "missing.tif")
    except OSError:
        links = False
    return root, links


@pytest.mark.parametrize("recursive", (True, False))
@pytest.mark.parametrize("name_filter", (None, "a", "c"))
def test_the_listing_keeps_the_candidates_rglob_found(tmp_path, recursive, name_filter):
    root, links = _tree(tmp_path)
    source = DirectorySourceSpec(root, recursive=recursive, suffixes=(".tif",), name_filter=name_filter)
    skip = (os.path.normcase(os.path.abspath(root / "processed")),)
    for skipped in ((), skip):
        found = output_preflight._directory_matching_paths(source, cancelled=lambda: False, skip=skipped)
        assert found == _reference(source, skipped)
    found = output_preflight._directory_matching_paths(source, cancelled=lambda: False)
    if name_filter is None:
        assert (root / "a.tif") in found and (root / "B.TIF") in found and (root / "folder.tif") not in found
        assert ((root / "scanA" / "c.tif") in found) is recursive
        assert (root / "scanA" / "deeper" / "d.tif") not in found
        if links:
            assert (root / "linked_file.tif") in found and (root / "dangling.tif") not in found
            assert (root / "linked_folder" / "f.tif") not in found


def test_a_missing_folder_lists_nothing_recursively_and_raises_otherwise(tmp_path):
    missing = tmp_path / "nowhere"
    recursive = DirectorySourceSpec(missing, recursive=True, suffixes=(".tif",))
    assert output_preflight._directory_matching_paths(recursive, cancelled=lambda: False) == _reference(recursive)
    flat = DirectorySourceSpec(missing, recursive=False, suffixes=(".tif",))
    with pytest.raises(FileNotFoundError):
        _reference(flat)
    with pytest.raises(FileNotFoundError):
        output_preflight._directory_matching_paths(flat, cancelled=lambda: False)
