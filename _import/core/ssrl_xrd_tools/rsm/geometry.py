from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import xrayutilities as xu


@dataclass(slots=True)
class DiffractometerConfig:
    """Geometry configuration for xu.QConversion and xu.HXRD."""

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

    def make_hxrd(self, energy: float) -> xu.HXRD:
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
