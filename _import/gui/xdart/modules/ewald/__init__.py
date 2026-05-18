"""Import surface for xdart live scan objects.

Note: this module historically also re-exported the legacy
``EwaldArch`` / ``EwaldSphere`` / ``ArchSeries`` aliases for one
release after the Live rename.  Those aliases were dropped after
the transitional release.  New code should always import from
:mod:`xdart.modules.live`; this module remains the in-package home
of the underlying classes.

Reader-side compatibility for ``.nxs`` files written by xdart 0.37.x
(which serialised the old class names into reduction provenance)
still works — see :mod:`xdart.modules.live_compat`.
"""

from .arch import LiveFrame
from .arch_series import LiveFrameSeries
from .sphere import LiveScan

__all__ = [
    "LiveFrame",
    "LiveScan",
    "LiveFrameSeries",
]
