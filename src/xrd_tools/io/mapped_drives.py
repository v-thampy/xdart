"""Windows mapped network drives: one folder spelled as a mapped drive (``X:\\bl11-3``) or as its share's network
path (``\\\\server\\share\\bl11-3``) is one place.

Maintainer, 2026-10-26: "x: maps to \\ssrlblfs1.slac.stanford.edu\\hardx. is there any way to detect that and suppress
this warning? ... if i use the open file dialog, it gives \\ssrl... but if i copy and paste from windows explorer - i
get x:. they should point to the same path in xdart"; his pick for how a field shows such a folder: "Drive letter,
X:\\... (Recommended)".

The mappings come from ``WNetGetConnectionW``. This is a synchronous provider call; Windows documents no timeout
guarantee for it. Off Windows there are no mappings and every function returns its input.
"""
from __future__ import annotations

import ntpath
import os
import string
import sys
import time

#: How long one read of the drive table is reused; a mapping changes rarely.
_CACHE_S = 10.0
#: WNetGetConnectionW: the mapping exists but its connection is down; the drive still names that share.
_ERROR_CONNECTION_UNAVAIL = 1201
_cache: tuple[float, dict[str, str]] | None = None


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _query_remotes() -> dict[str, str]:
    """``{"X:": "\\\\server\\share"}`` for every drive letter mapped to a network folder (Windows only)."""
    import ctypes
    from ctypes import wintypes

    get = ctypes.WinDLL("mpr").WNetGetConnectionW
    get.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD))
    get.restype = wintypes.DWORD
    present = ctypes.WinDLL("kernel32").GetLogicalDrives()
    remotes = {}
    for index, letter in enumerate(string.ascii_uppercase):
        if not present & (1 << index):
            continue
        size = wintypes.DWORD(1024)
        buffer = ctypes.create_unicode_buffer(size.value)
        status = get(f"{letter}:", buffer, ctypes.byref(size))
        if status in (0, _ERROR_CONNECTION_UNAVAIL) and buffer.value.startswith("\\\\"):
            remotes[f"{letter}:"] = ntpath.normpath(buffer.value)
    return remotes


def drive_remotes() -> dict[str, str]:
    """The mapped drives, read at most every few seconds; ``{}`` off Windows or when the table cannot be read."""
    global _cache
    if not _is_windows():
        return {}
    now = time.monotonic()
    if _cache is not None and now - _cache[0] < _CACHE_S:
        return _cache[1]
    try:
        remotes = _query_remotes()
    except (OSError, AttributeError, ValueError):
        remotes = {}
    _cache = (now, remotes)
    return remotes


def _network_path(path: str) -> str | None:
    """``path`` with backslashes when it is a share's network path; None otherwise (``\\\\?\\`` and ``\\\\.\\``
    device paths are not shares)."""
    spelled = ntpath.normpath(path)
    if not spelled.startswith("\\\\") or spelled.startswith(("\\\\?\\", "\\\\.\\")):
        return None
    return spelled


def as_mapped_drive(path: str, remotes: dict[str, str] | None = None) -> str:
    """``path`` spelled with the drive letter mapped to its share (the longest mapped folder wins), or ``path`` itself
    when it is no network path or no drive maps it."""
    remotes = drive_remotes() if remotes is None else remotes
    spelled = _network_path(path) if path and remotes else None
    if spelled is None:
        return path
    folded = ntpath.normcase(spelled)
    best: tuple[str, str] | None = None
    for drive, remote in remotes.items():
        prefix = ntpath.normpath(remote)
        key = ntpath.normcase(prefix)
        if (folded == key or folded.startswith(key + "\\")) and (best is None or len(prefix) > len(best[1])):
            best = (drive, prefix)
    if best is None:
        return path
    drive, prefix = best
    return drive + (spelled[len(prefix):] or "\\")


def as_network_path(path: str, remotes: dict[str, str] | None = None) -> str:
    """``path`` spelled as its share's network path when it starts with a mapped drive letter, or ``path`` itself.
    For comparing two spellings, never for showing one."""
    remotes = drive_remotes() if remotes is None else remotes
    if not path or not remotes:
        return path
    drive, rest = ntpath.splitdrive(ntpath.normpath(path))
    remote = remotes.get(drive.upper()) if len(drive) == 2 and drive.endswith(":") else None
    if remote is None:
        return path
    return ntpath.normpath(remote) + (rest if not rest or rest.startswith("\\") else "\\" + rest)


def relative_within(src: str, root: str, *, pathmod=None, remotes: dict[str, str] | None = None) -> str | None:
    """The POSIX relative path of absolute ``src`` inside absolute ``root``, or None when it lies outside: compared as
    spelled, then, if either names a mapped drive, with both spelled as network paths. ``pathmod``: the path module,
    ``os.path`` as it is when called (tests pass ``ntpath``, or patch ``os.path``)."""
    pathmod = os.path if pathmod is None else pathmod
    remotes = drive_remotes() if remotes is None else remotes
    candidates = [(src, root)]
    if remotes:
        network = (as_network_path(src, remotes), as_network_path(root, remotes))
        if network != (src, root):
            candidates.append(network)
    for left, right in candidates:
        try:
            inside = pathmod.normcase(pathmod.commonpath([left, right])) == pathmod.normcase(right)
        except ValueError:  # different drives, or a drive against a share (Windows)
            inside = False
        if inside:
            return pathmod.relpath(left, right).replace(pathmod.sep, "/")
    return None


__all__ = ["as_mapped_drive", "as_network_path", "drive_remotes", "relative_within"]
