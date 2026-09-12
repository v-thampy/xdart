"""The one platform fact behind every descriptor-vs-pathname stat compare.

CPython on Windows fills ``st_ctime`` from the change time for a handle
``fstat`` but from the creation time for a pathname ``stat``/``lstat``
(NTFS, py3.13: the two views of one untouched file differ by the
create-to-write gap), so an identity that compares a descriptor view
against a pathname view can never agree there.  Every such comparison in
the tree routes its ctime slot through :func:`identity_ctime_ns`: exact on
POSIX, where ctime still catches a same-size same-mtime in-place edit, and
neutral on win32, where the content digest remains the authority for that
case.  Receipts keep recording the observed ctime as evidence; only the
comparison slot is neutral.
"""

from __future__ import annotations

import sys

IDENTITY_CARRIES_CTIME = sys.platform != "win32"


def identity_ctime_ns(ctime_ns: int) -> int:
    """The comparable ctime slot for *ctime_ns*: itself on POSIX, ``0`` on win32."""
    return int(ctime_ns) if IDENTITY_CARRIES_CTIME else 0
