"""Low-level geometry configuration primitives.

This module lives in :mod:`ssrl_xrd_tools.core` so that
:class:`DiffractometerConfig` can be referenced from other ``core``
modules (notably :mod:`ssrl_xrd_tools.core.config`) without forcing
``core`` to depend on the :mod:`ssrl_xrd_tools.rsm` layer.

:class:`DiffractometerConfig` is a pure dataclass. Its
:meth:`make_hxrd` method uses ``xrayutilities``, which is imported
lazily so that users who never call it pay no import cost and
``core`` does not grow a hard dependency on ``xrayutilities`` at
import time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # Only needed for type checkers; avoided at runtime.
    import xrayutilities as xu


@dataclass(slots=True)
class DiffractometerConfig:
    """Geometry configuration for ``xu.QConversion`` and ``xu.HXRD``."""

    sample_rot: tuple[str, ...] = ("z-", "y+", "z-")
    detector_rot: tuple[str, ...] = ("z-",)
    r_i: tuple[float, float, float] = (0.0, 1.0, 0.0)

    q_conv_kwargs: dict[str, Any] = field(default_factory=dict)

    hxrd_n: tuple[float, float, float] = (0.0, 1.0, 0.0)
    hxrd_q: tuple[float, float, float] = (1.0, 0.0, 0.0)
    hxrd_geometry: str = "real"
    hxrd_kwargs: dict[str, Any] = field(default_factory=dict)

    init_area_detrot: str = "z-"
    init_area_tiltazimuth: str = "x+"
    ang2q_kwargs: dict[str, Any] = field(default_factory=dict)

    def make_hxrd(self, energy: float) -> "xu.HXRD":
        """Build an ``xu.HXRD`` instance at the given energy.

        ``xrayutilities`` is imported lazily inside this method so that
        importing :mod:`ssrl_xrd_tools.core` never triggers the xu
        import (it is still a declared dependency and must be
        installed before calling this method).
        """
        import xrayutilities as xu  # noqa: PLC0415 — intentional lazy import

        qconversion = xu.QConversion(
            self.sample_rot,
            self.detector_rot,
            self.r_i,
            **self.q_conv_kwargs,
        )
        return xu.HXRD(
            self.hxrd_n,
            self.hxrd_q,
            geometry=self.hxrd_geometry,
            en=energy,
            qconv=qconversion,
            **self.hxrd_kwargs,
        )


__all__ = ["DiffractometerConfig"]
