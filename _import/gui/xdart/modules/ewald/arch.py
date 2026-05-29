"""Deprecated module path — use :mod:`xdart.modules.ewald.frame`.

Thin shim kept after the sphere/arch → scan/frame rename so older
import sites and v2 ``.nxs`` provenance strings referencing
``xdart.modules.ewald.arch.LiveFrame`` (or the legacy ``EwaldArch``
class via :mod:`xdart.modules.live_compat`) still resolve.
"""
from xdart.modules.ewald.frame import LiveFrame, _make_thumbnail

# Legacy class alias (pre-Live rename name).
EwaldArch = LiveFrame

__all__ = ["LiveFrame", "EwaldArch", "_make_thumbnail"]
