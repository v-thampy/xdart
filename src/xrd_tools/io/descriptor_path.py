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

:func:`created_leaf_misplacement` reads it once and holds the directory
it names to the identity of the directory the installer inspected, and
the entry it names to the requested one.  Renaming that directory keeps
its identity; substituting it does not.  The check is identity-based on
purpose: a pathname compare would have to reconcile short names, case
and ``\\\\?\\`` prefixes, and a lexical ``realpath`` of the target
computed after the exchange would follow the very link it is meant to
catch.

A leaf that landed elsewhere is disposed of through the handle that
created it (:func:`dispose_created_leaf`), never by name: between a
by-name inspection and a by-name unlink the entry can be replaced by a
foreign file, and the parent can be put back so the misplaced object is
no longer even reachable by the name (Codex review of 0ed7a46c, F2).
Windows, the only host whose production install is by name, deletes
through the handle (``FileDispositionInfo``); POSIX has no unlink by
descriptor, so there the caller refuses and names the leaf it left.
"""

from __future__ import annotations

import errno
import os
import stat
import sys
from pathlib import Path

# FILE_NAME_NORMALIZED | VOLUME_NAME_DOS
_WIN32_FINAL_PATH_FLAGS = 0
_DARWIN_MAXPATHLEN = 1024

_GENERIC_WRITE = 0x40000000
_DELETE = 0x00010000
_FILE_SHARE_READ = 0x00000001
_CREATE_NEW = 1
_FILE_ATTRIBUTE_NORMAL = 0x00000080
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_ERROR_FILE_EXISTS = 80
_ERROR_ALREADY_EXISTS = 183
# SetFileInformationByHandle information classes.
_FILE_DISPOSITION_INFO = 4
_FILE_DISPOSITION_INFO_EX = 21
_FILE_DISPOSITION_DELETE = 0x00000001
_FILE_DISPOSITION_POSIX_SEMANTICS = 0x00000002


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


def _win32_kernel32():
    import ctypes

    return ctypes.WinDLL("kernel32", use_last_error=True)


def _win32_final_path(descriptor: int) -> str:
    import ctypes
    from ctypes import wintypes
    import msvcrt

    function = _win32_kernel32().GetFinalPathNameByHandleW
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


def create_exclusive_leaf(path: str) -> int:
    """Create the regular file *path* by name, exclusively, for writing.

    Returns a descriptor the caller writes through and closes.  A name
    that already exists -- a symbolic link included -- raises
    ``FileExistsError``.  On Windows the handle is opened with ``DELETE``
    access as well, so :func:`dispose_created_leaf` can remove the object
    through it; the CRT ``os.open`` never asks for that right.
    """
    if sys.platform == "win32":
        return _win32_create_exclusive(path)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    return os.open(path, flags, 0o644)


def _win32_create_exclusive(path: str) -> int:
    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel32 = _win32_kernel32()
    function = kernel32.CreateFileW
    function.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    )
    function.restype = wintypes.HANDLE
    invalid = wintypes.HANDLE(-1).value
    handle = function(
        os.fspath(path),
        _GENERIC_WRITE | _DELETE,
        _FILE_SHARE_READ,
        None,
        _CREATE_NEW,
        # A reparse point at the leaf name is the existing entry, not a
        # path to create through: CREATE_NEW then fails as "exists".
        _FILE_ATTRIBUTE_NORMAL | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle == invalid:
        code = ctypes.get_last_error()
        error = ctypes.WinError(code)
        if code in (_ERROR_FILE_EXISTS, _ERROR_ALREADY_EXISTS) and not isinstance(
            error, FileExistsError
        ):
            raise FileExistsError(errno.EEXIST, error.strerror, path) from error
        raise error
    try:
        return msvcrt.open_osfhandle(
            handle, os.O_WRONLY | os.O_BINARY | os.O_NOINHERIT
        )
    except OSError:
        kernel32.CloseHandle(wintypes.HANDLE(handle))
        raise


def directory_identity(value: os.stat_result) -> tuple[int, int]:
    """The (st_dev, st_ino) of one inspected directory."""
    return (int(value.st_dev), int(value.st_ino))


# The slots of a chain state: (st_mode, st_dev, st_ino, st_size,
# st_mtime_ns, seamed st_ctime_ns), as the no-follow chain walkers record
# them.  A directory is held to the first three: creating or removing a
# sibling entry beside the chain advances its mtime (and its size on some
# filesystems) without the chain having moved.
_CHAIN_STATE_FIELDS = ("mode", "dev", "ino", "size", "mtime_ns", "ctime_ns")
_DIRECTORY_HELD = slice(0, 3)


def chain_drift(
    opened: tuple[tuple[int, ...], ...],
    current: tuple[tuple[int, ...], ...],
    components: tuple[str, ...],
) -> str | None:
    """``None`` while the *current* chain still names the *opened* one;
    otherwise the first component that moved, with the field that did.

    Every ancestor directory is held to its identity (mode, device, inode)
    only, so a concurrent write beside the chain is not a change of the
    chain; the leaf (the last slot) is held to its full recorded state.
    Holding the ancestors' mtime refused a capture under the runner's
    system temp directory whenever another process created an entry there
    (PR #1 round 12, macos-15-intel), and named nothing.
    """
    if len(components) != len(opened):
        return f"chain of {len(opened)} slots names {len(components)} components"
    if len(opened) != len(current):
        return f"chain length {len(opened)} -> {len(current)}"
    last = len(opened) - 1
    for index, (before, after) in enumerate(zip(opened, current)):
        held = slice(None) if index == last else _DIRECTORY_HELD
        for name, x, y in zip(_CHAIN_STATE_FIELDS[held], before[held], after[held]):
            if x != y:
                kind = "leaf" if index == last else "ancestor"
                return f"{kind} {components[index]}: {name} {x} -> {y}"
    return None


def chain_components(project: str, relative: str) -> tuple[str, ...]:
    """The path each slot of a chain state names: the root, then every
    component of *project* below it, then every component of *relative*."""
    project_parts = Path(project).parts
    components = [project_parts[0]]
    current = project_parts[0]
    for part in project_parts[1:] + Path(relative).parts:
        current = os.path.join(current, part)
        components.append(current)
    return tuple(components)


def created_leaf_misplacement(
    descriptor: int, parent_identity: tuple[int, int], name: str,
) -> str | None:
    """``None`` when the object behind *descriptor* is the entry *name*
    directly inside the directory whose :func:`directory_identity` is
    *parent_identity*; otherwise the reason it is not, naming the final
    path where the platform gives one.

    Unreadable final paths count as misplaced: the caller refuses rather
    than guesses.
    """
    try:
        final = descriptor_final_path(descriptor)
    except OSError as error:
        return f"final path of the created leaf is unavailable ({error})"
    try:
        landing = os.lstat(os.path.dirname(final))
    except OSError as error:
        return f"{final}: landing directory is unavailable ({error})"
    if not stat.S_ISDIR(landing.st_mode) or (
        directory_identity(landing) != tuple(parent_identity)
    ):
        return f"{final}: outside the inspected directory"
    if os.path.basename(final) != name:
        return f"{final}: not the requested entry"
    return None


def dispose_created_leaf(descriptor: int) -> bool:
    """Delete the object behind *descriptor* through the handle itself,
    wherever it landed and whatever the name now resolves to.

    True when the object is gone (or goes with the last close); False
    where the platform cannot delete by descriptor (every POSIX host --
    there the by-name installer runs only under test) or refuses, so the
    caller reports the leaf it leaves.  Never touches a name.
    """
    if sys.platform != "win32":
        return False
    try:
        _win32_dispose(descriptor)
    except OSError:
        return False
    return True


def _win32_dispose(descriptor: int) -> None:
    import ctypes
    from ctypes import wintypes
    import msvcrt

    function = _win32_kernel32().SetFileInformationByHandle
    function.argtypes = (
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
    )
    function.restype = wintypes.BOOL
    handle = msvcrt.get_osfhandle(descriptor)
    # FILE_DISPOSITION_INFO_EX: the name goes now, POSIX-style (Windows 10
    # 1709+); FILE_DISPOSITION_INFO: the object goes with the last close.
    flags = wintypes.DWORD(_FILE_DISPOSITION_DELETE | _FILE_DISPOSITION_POSIX_SEMANTICS)
    if function(handle, _FILE_DISPOSITION_INFO_EX, ctypes.byref(flags), ctypes.sizeof(flags)):
        return
    delete = ctypes.c_ubyte(1)
    if function(handle, _FILE_DISPOSITION_INFO, ctypes.byref(delete), ctypes.sizeof(delete)):
        return
    raise ctypes.WinError(ctypes.get_last_error())
