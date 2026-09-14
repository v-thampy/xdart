"""The one platform fact behind every descriptor-vs-pathname stat compare.

CPython on Windows fills ``st_ctime`` from the change time for a handle
``fstat`` but from the creation time for a pathname ``stat``/``lstat``
(NTFS, py3.13: the two views of one untouched file differ by the
create-to-write gap), so an identity that compares a descriptor view
against a pathname view can never agree there.  Every such comparison in
the tree routes its ctime slot through :func:`identity_ctime_ns`: exact on
POSIX, where ctime still catches a same-size same-mtime in-place edit, and
neutral on win32.

A comparison with a DESCRIPTOR view on both sides (two ``fstat`` views of
one open file, or a descriptor view against a receipt that recorded one)
never goes through the seam: it keeps the raw ctime on every platform,
because win32 fills the handle's ``st_ctime`` from NTFS ChangeTime, which
every write and every ``utime`` advance.  That is what lets the stat-only
revalidators hand out a recorded digest without rereading the bytes and
lets a hash bracketed by two descriptor views refuse a same-size
same-mtime rewrite inside its window, on win32 too.  Receipts therefore
always record the descriptor's observed ctime, never a pathname stat's.
"""

from __future__ import annotations

import sys

IDENTITY_CARRIES_CTIME = sys.platform != "win32"


def identity_ctime_ns(ctime_ns: int) -> int:
    """The comparable ctime slot for *ctime_ns*: itself on POSIX, ``0`` on win32."""
    return int(ctime_ns) if IDENTITY_CARRIES_CTIME else 0
