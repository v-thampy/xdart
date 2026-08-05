"""Detached, filesystem-only catalog values for processed scan browsing."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat
import threading
import time

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


class DirectoryModifiedCache:
    """Bound shallow directory timestamp reads for periodic Date refreshes."""

    def __init__(self, *, ttl_s: float = 2.0, max_entries: int = 512) -> None:
        self._ttl_s = float(ttl_s)
        self._max_entries = int(max_entries)
        if self._ttl_s < 0.0 or self._max_entries < 1:
            raise ValueError("directory timestamp cache bounds are invalid")
        self._lock = threading.Lock()
        self._generation = 0
        self._entries: dict[str, tuple[int, float, int]] = {}

    def clear(self) -> None:
        """Invalidate every cached observation without racing an active scan."""

        with self._lock:
            self._generation += 1
            self._entries.clear()

    def modified_ns(self, directory: Path, own_mtime_ns: int) -> int:
        path = os.path.normcase(os.path.abspath(os.fspath(directory)))
        now = time.monotonic()
        with self._lock:
            cached = self._entries.get(path)
            if cached is not None:
                cached_own_mtime_ns, checked_at, effective_mtime_ns = cached
                if (
                    cached_own_mtime_ns == own_mtime_ns
                    and now - checked_at < self._ttl_s
                ):
                    return effective_mtime_ns
            generation = self._generation

        effective_mtime_ns, cacheable = _directory_modified_ns(
            directory,
            own_mtime_ns,
        )
        if not cacheable:
            return effective_mtime_ns
        with self._lock:
            # An explicit refresh that landed while scandir was running owns
            # the newer generation; never repopulate it with the old read.
            if generation != self._generation:
                return effective_mtime_ns
            if len(self._entries) >= self._max_entries:
                self._entries.pop(next(iter(self._entries)), None)
            self._entries[path] = (
                own_mtime_ns,
                now,
                effective_mtime_ns,
            )
        return effective_mtime_ns


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
    *,
    inspect_directory_contents: bool = False,
    directory_time_cache: DirectoryModifiedCache | None = None,
) -> tuple[BrowserCatalogEntry, ...]:
    """Enumerate regular NeXus artifacts without opening their contents."""

    if type(directory) is not str or not directory:
        return ()
    if type(inspect_directory_contents) is not bool:
        raise TypeError("directory-content timestamp policy must be boolean")
    if (
        directory_time_cache is not None
        and type(directory_time_cache) is not DirectoryModifiedCache
    ):
        raise TypeError("directory timestamp cache must be exact")
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
            modified_ns = int(fact.st_mtime_ns)
            if inspect_directory_contents:
                if directory_time_cache is None:
                    modified_ns, _cacheable = _directory_modified_ns(
                        child,
                        modified_ns,
                    )
                else:
                    modified_ns = directory_time_cache.modified_ns(
                        child,
                        modified_ns,
                    )
            directories.append(BrowserCatalogEntry(
                os.path.abspath(os.fspath(child)),
                f"{child.name}/",
                modified_ns,
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


def _directory_modified_ns(
    directory: Path,
    own_mtime_ns: int,
) -> tuple[int, bool]:
    """Return newest immediate-child mtime and whether the read is cacheable."""

    effective_mtime_ns: int | None = None
    try:
        with os.scandir(directory) as children:
            for child in children:
                try:
                    child_mtime_ns = int(
                        child.stat(follow_symlinks=False).st_mtime_ns
                    )
                except OSError:
                    continue
                effective_mtime_ns = (
                    child_mtime_ns
                    if effective_mtime_ns is None
                    else max(effective_mtime_ns, child_mtime_ns)
                )
    except OSError:
        # Unreadable/transient directories retain their own useful timestamp,
        # but the failed observation is not cached.
        return own_mtime_ns, False
    return (
        own_mtime_ns if effective_mtime_ns is None else effective_mtime_ns,
        True,
    )


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
    "DirectoryModifiedCache",
    "enumerate_processed_artifacts",
    "natural_name_key",
    "processed_directory",
]
