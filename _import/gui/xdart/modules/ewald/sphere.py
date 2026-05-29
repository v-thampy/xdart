"""Deprecated module path — use :mod:`xdart.modules.ewald.scan`.

Kept as a thin shim after the sphere/arch → scan/frame rename so that
older import sites and v2 ``.nxs`` provenance strings that reference
``xdart.modules.ewald.sphere.LiveScan`` (or the legacy ``EwaldSphere``
class via :mod:`xdart.modules.live_compat`) still resolve.
"""
from xdart.modules.ewald.scan import LiveScan

# Legacy class alias (pre-Live rename name).
EwaldSphere = LiveScan

__all__ = ["LiveScan", "EwaldSphere"]
