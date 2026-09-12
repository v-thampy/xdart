"""Where an open descriptor's object lives, for the no-dir_fd installers.

A by-name ``O_CREAT|O_EXCL`` open proves nothing about WHERE the leaf was
created: a parent directory exchanged for a symbolic link between the
installer's lstat inspection and the open lands the new file inside the
link's target (Codex PR #1 review, F2).  The kernel does know where the
object it handed back lives, and each platform exposes that final path
for an open handle:

* win32: ``GetFinalPathNameByHandleW`` (the call ``ntpath.realpath`` uses);
* linux: the ``/proc/self/fd/N`` link;
* darwin: ``fcntl(F_GETPATH)``.

:func:`leaf_created_inside` reads it once and holds the directory it
names to the identity of the directory the installer inspected.  Renaming
that directory keeps its identity; substituting it does not.  The check is
identity-based on purpose: a pathname compare would have to reconcile
short names, case and ``\\\\?\\`` prefixes, and a lexical ``realpath`` of
the target computed after the exchange would follow the very link it is
meant to catch.
"""

from __future__ import annotations

import errno
import os
import stat
import sys

# FILE_NAME_NORMALIZED | VOLUME_NAME_DOS
_WIN32_FINAL_PATH_FLAGS = 0
_DARWIN_MAXPATHLEN = 1024


def descriptor_final_path(descriptor: int) -> str:
    """The kernel's final pathname of the object behind *descriptor*.

    Raises ``OSError`` where the platform cannot answer (no ``/proc``, an
    unlinked object, an unsupported platform) so callers refuse rather
    than guess.
    """
    if sys.platform == "win32":
        return _win32_final_path(descriptor)
    if sys.platform == "linux":
        return os.readlink(f"/proc/self/fd/{int(descriptor)}")
    if sys.platform == "darwin":
        import fcntl

        raw = fcntl.fcntl(descriptor, fcntl.F_GETPATH, bytes(_DARWIN_MAXPATHLEN))
        return os.fsdecode(raw.split(b"\0", 1)[0])
    raise OSError(
        errno.ENOSYS, "descriptor final path is not available on this platform",
    )


def _win32_final_path(descriptor: int) -> str:
    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel32.GetFinalPathNameByHandleW
    function.argtypes = (
        wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD,
    )
    function.restype = wintypes.DWORD
    handle = msvcrt.get_osfhandle(descriptor)
    size = 1024
    while True:
        buffer = ctypes.create_unicode_buffer(size)
        length = function(handle, buffer, size, _WIN32_FINAL_PATH_FLAGS)
        if length == 0:
            raise ctypes.WinError(ctypes.get_last_error())
        if length < size:
            # Possibly ``\\?\``-prefixed; every os.* call below accepts that.
            return buffer.value
        size = length + 1


def directory_identity(value: os.stat_result) -> tuple[int, int]:
    """The (st_dev, st_ino) of one inspected directory."""
    return (int(value.st_dev), int(value.st_ino))


def leaf_created_inside(descriptor: int, parent_identity: tuple[int, int]) -> bool:
    """Whether the object behind *descriptor* lives directly inside the
    directory whose :func:`directory_identity` is *parent_identity*.

    False whenever the final path cannot be read or names a directory
    other than the inspected one; the caller refuses and removes the
    misplaced empty leaf (:func:`unlink_empty_created_leaf`).
    """
    try:
        final = descriptor_final_path(descriptor)
        landing = os.lstat(os.path.dirname(final))
    except OSError:
        return False
    return stat.S_ISDIR(landing.st_mode) and (
        directory_identity(landing) == tuple(parent_identity)
    )


def unlink_empty_created_leaf(path: str, identity: tuple[int, int]) -> None:
    """Remove the empty regular file the caller just created at *path*, and
    only that object: the name must still carry *identity* and hold no
    bytes.  Best effort; a failure leaves the leaf for the caller's
    refusal to report."""
    try:
        named = os.lstat(path)
    except OSError:
        return
    if (
        stat.S_ISREG(named.st_mode)
        and int(named.st_size) == 0
        and directory_identity(named) == tuple(identity)
    ):
        try:
            os.unlink(path)
        except OSError:
            pass
