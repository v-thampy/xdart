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

On a Windows share the SMB client also serves a pathname stat's size and
mtime from its attribute cache (about ten seconds by default), while an open
handle sees the file as it is.  While another computer appends to a SPEC
file, the two views of one file therefore disagree for as long as the scan
records (beamline 2026-11-27: every RSM Preview during a scan refused
SOURCE_REVISION_CHANGED, while the same data previewed on a Mac).  Where a
pathname view only has to show that the name still reaches the file a
descriptor read, :data:`PATHNAME_STATE_IS_CURRENT` keeps the size and mtime
in that compare on POSIX and leaves the file object (device, inode) alone
deciding it on win32.  The bytes themselves are proved through the
descriptor.
"""

from __future__ import annotations

import sys

IDENTITY_CARRIES_CTIME = sys.platform != "win32"

#: Whether a pathname stat's size and mtime are the file's current ones (POSIX), or may lag an open handle's
#: (win32: a share's attribute cache).
PATHNAME_STATE_IS_CURRENT = sys.platform != "win32"


def identity_ctime_ns(ctime_ns: int) -> int:
    """The comparable ctime slot for *ctime_ns*: itself on POSIX, ``0`` on win32."""
    return int(ctime_ns) if IDENTITY_CARRIES_CTIME else 0
