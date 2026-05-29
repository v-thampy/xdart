"""Deprecated module path — use :mod:`xdart.modules.ewald.frame_series`.

Thin shim kept after the sphere/arch → scan/frame rename so older
import sites and v2 ``.nxs`` provenance strings referencing
``xdart.modules.ewald.arch_series.*`` (or the legacy ``ArchSeries``
class via :mod:`xdart.modules.live_compat`) still resolve.
"""
from xdart.modules.ewald.frame_series import (
    LiveFrameSeries,
    _IndexedList,
    _load_frame_v2,
)

# Legacy aliases (pre-rename names).
ArchSeries = LiveFrameSeries
_load_arch_v2 = _load_frame_v2

__all__ = ["LiveFrameSeries", "ArchSeries", "_IndexedList", "_load_frame_v2"]
