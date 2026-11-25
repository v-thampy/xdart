"""The one platform fact behind every descriptor-vs-pathname stat compare.

CPython on Windows fills ``st_ctime`` from the change time for a handle
``fstat`` but from the creation time for a pathname ``stat``/``lstat``
(NTFS, py3.13: the two views of one untouched file differ by the
create-to-write gap), so the ctime fields can disagree for the same
unchanged file.  Mixed-view comparisons route the slot through
:func:`identity_ctime_ns`: exact on
POSIX, where ctime still catches a same-size same-mtime in-place edit, and
neutral on win32.

A comparison with a DESCRIPTOR view on both sides (two ``fstat`` views of
one open file, or a descriptor view against a receipt that recorded one)
never goes through the seam: it keeps the raw ctime on every platform,
so a ChangeTime advance visible to a Windows handle can still invalidate
a recorded digest or a bracketed read.  Windows may expose the same
descriptor timestamps across a concurrent same-size write.  In that case
a SPEC occurrence check can accept the bytes it already read, and a fresh
check rereads and refuses a persistent edit.  Raw descriptor revision
receipts record the descriptor's observed ctime.

Live and Append also compare an already-processed source file's pathname
views across time (the Live prefix proof every ten seconds, the published
metadata fence, the Append source digest; ``execution_graph._settled``).  On a
Windows share the creation time a pathname stat reports can settle
milliseconds to a third of a second after the file is written (beamline
2026-10-25), so those comparisons route the slot through the seam too.  The
capture (``SourceFileState``) stays exact for its other consumers.  On win32
the slot never caught an in-place edit; size and mtime still do.
"""

from __future__ import annotations

import sys

IDENTITY_CARRIES_CTIME = sys.platform != "win32"


def identity_ctime_ns(ctime_ns: int) -> int:
    """The comparable ctime slot for *ctime_ns*: itself on POSIX, ``0`` on win32."""
    return int(ctime_ns) if IDENTITY_CARRIES_CTIME else 0
