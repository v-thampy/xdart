"""Detached, filesystem-only catalog values for processed scan browsing."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat

from xrd_tools.io import is_readable_output_path


@dataclass(frozen=True, slots=True)
class BrowserCatalogEntry:
    """One detached processed artifact or navigable directory."""

    artifact: str
    label: str
    modified_ns: int
    is_directory: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.artifact) is not str
            or not self.artifact
            or type(self.label) is not str
            or not self.label
            or type(self.modified_ns) is not int
            or self.modified_ns < 0
            or type(self.is_directory) is not bool
        ):
            raise TypeError("browser catalog entry is invalid")


def processed_directory(save_path: str) -> str:
    """Return the directory that owns the configured processed artifacts."""

    if type(save_path) is not str or not save_path:
        return ""
    requested = Path(save_path).expanduser()
    directory = (
        requested.parent
        if is_readable_output_path(requested)
        else requested
    )
    return os.path.abspath(os.fspath(directory))


def enumerate_processed_artifacts(
    directory: str,
) -> tuple[BrowserCatalogEntry, ...]:
    """Enumerate regular NeXus artifacts without opening their contents."""

    if type(directory) is not str or not directory:
        return ()
    root = Path(os.path.abspath(os.path.expanduser(directory)))
    try:
        children = tuple(root.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        parent_entry = _parent_navigation_entry(root)
        return () if parent_entry is None else (parent_entry,)
    except OSError:
        return ()
    directories: list[BrowserCatalogEntry] = []
    artifacts: list[BrowserCatalogEntry] = []
    parent_entry = _parent_navigation_entry(root)
    if parent_entry is not None:
        directories.append(parent_entry)
    for child in children:
        try:
            fact = child.stat()
        except OSError:
            continue
        if child.is_dir():
            directories.append(BrowserCatalogEntry(
                os.path.abspath(os.fspath(child)),
                f"{child.name}/",
                fact.st_mtime_ns,
                True,
            ))
            continue
        if (
            not is_readable_output_path(child)
            or not child.is_file()
        ):
            continue
        artifacts.append(
            BrowserCatalogEntry(
                os.path.abspath(os.fspath(child)),
                child.name,
                fact.st_mtime_ns,
            )
        )
    parent_entries = tuple(
        entry for entry in directories if entry.label == ".."
    )
    entries = sorted(
        (
            *(entry for entry in directories if entry.label != ".."),
            *artifacts,
        ),
        key=lambda entry: natural_name_key(
            entry.label.removesuffix("/")
            if entry.is_directory
            else entry.label
        ),
    )
    return (*parent_entries, *entries)


def _parent_navigation_entry(
    root: Path,
) -> BrowserCatalogEntry | None:
    parent = root.parent
    if parent == root:
        return None
    try:
        parent_fact = parent.stat()
    except OSError:
        return None
    if not stat.S_ISDIR(parent_fact.st_mode):
        return None
    return BrowserCatalogEntry(
        os.fspath(parent),
        "..",
        parent_fact.st_mtime_ns,
        True,
    )


def natural_name_key(value: str) -> tuple[tuple[int, object], ...]:
    """Return a case-insensitive key with numeric components ordered as ints."""

    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


__all__ = [
    "BrowserCatalogEntry",
    "enumerate_processed_artifacts",
    "natural_name_key",
    "processed_directory",
]
