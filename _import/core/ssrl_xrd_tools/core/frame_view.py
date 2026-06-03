"""Canonical per-frame display/round-trip records.

These types are intentionally GUI-free.  They describe the data needed to
display, validate, stitch, fit, or round-trip one processed detector frame
without depending on xdart's live objects or Qt state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


class TwoDKind(str, Enum):
    """Identity of the two axes in a 2D integrated image."""

    Q_CHI = "q_chi"
    QIP_QOOP = "qip_qoop"
    QTOT_CHIGI = "qtot_chigi"
    EXIT_ANGLES = "exit_angles"


@dataclass(frozen=True, slots=True)
class Axis:
    """One display/data axis.

    ``label`` and ``unit`` are the stable identity used by display code.
    ``values`` is optional so the same type can describe a plot label or a
    concrete sampled axis.
    """

    label: str
    unit: str = ""
    log: bool = False
    values: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.values is None:
            return
        values = np.asarray(self.values, dtype=float).copy()
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class FrameGeometry:
    """Per-frame geometry needed by GI, stitching, and RSM pipelines."""

    rot1: float | None = None
    rot2: float | None = None
    rot3: float | None = None
    incident_angle: float | None = None
    poni: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.poni is not None:
            object.__setattr__(self, "poni", MappingProxyType(dict(self.poni)))


def _readonly_array(value: Any, *, dtype: Any = float) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=dtype).copy()
    arr.setflags(write=False)
    return arr


def _readonly_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not value:
        return MappingProxyType({})
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class FrameView:
    """Immutable display-ready view of one processed frame.

    The 2D intensity convention is always ``(axis_2d_y, axis_2d_x)``.  This
    matches the stacked NeXus reader shape ``(chi, q)`` for standard cakes and
    ``(qoop, qip)`` for GI qip/qoop cakes.
    """

    label: int | str
    axis_1d: Axis | None = None
    intensity_1d: np.ndarray | None = None
    sigma_1d: np.ndarray | None = None
    axis_2d_x: Axis | None = None
    axis_2d_y: Axis | None = None
    intensity_2d: np.ndarray | None = None
    sigma_2d: np.ndarray | None = None
    two_d_kind: TwoDKind = TwoDKind.Q_CHI
    raw: np.ndarray | None = None
    thumbnail: np.ndarray | None = None
    mask_baked: bool = False
    metadata_raw: Mapping[str, Any] = field(default_factory=dict)
    metadata_numeric: Mapping[str, float] = field(default_factory=dict)
    incident_angle: float | None = None
    geometry: FrameGeometry | None = None
    source_path: str | None = None
    source_frame_index: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "intensity_1d", _readonly_array(self.intensity_1d))
        object.__setattr__(self, "sigma_1d", _readonly_array(self.sigma_1d))
        object.__setattr__(self, "intensity_2d", _readonly_array(self.intensity_2d))
        object.__setattr__(self, "sigma_2d", _readonly_array(self.sigma_2d))
        object.__setattr__(self, "raw", _readonly_array(self.raw))
        object.__setattr__(self, "thumbnail", _readonly_array(self.thumbnail))
        object.__setattr__(self, "metadata_raw", _readonly_mapping(self.metadata_raw))
        object.__setattr__(self, "metadata_numeric", _readonly_mapping(self.metadata_numeric))
        object.__setattr__(self, "extra", _readonly_mapping(self.extra))
        if not isinstance(self.two_d_kind, TwoDKind):
            object.__setattr__(self, "two_d_kind", TwoDKind(str(self.two_d_kind)))
        self._validate_shapes()

    @property
    def has_1d(self) -> bool:
        return self.axis_1d is not None and self.intensity_1d is not None

    @property
    def has_2d(self) -> bool:
        return (
            self.axis_2d_x is not None
            and self.axis_2d_y is not None
            and self.intensity_2d is not None
        )

    @property
    def scan_info(self) -> Mapping[str, Any]:
        """Compatibility spelling for existing notebook/display vocabulary."""
        return self.metadata_raw

    def _validate_shapes(self) -> None:
        if self.axis_1d is not None and self.intensity_1d is not None:
            if self.axis_1d.values is not None and self.axis_1d.values.shape != self.intensity_1d.shape:
                raise ValueError(
                    f"1D axis shape {self.axis_1d.values.shape} != "
                    f"intensity shape {self.intensity_1d.shape}"
                )
            if self.sigma_1d is not None and self.sigma_1d.shape != self.intensity_1d.shape:
                raise ValueError(
                    f"1D sigma shape {self.sigma_1d.shape} != "
                    f"intensity shape {self.intensity_1d.shape}"
                )
        if self.axis_2d_x is not None and self.axis_2d_y is not None and self.intensity_2d is not None:
            ny, nx = self.intensity_2d.shape
            if self.axis_2d_x.values is not None and self.axis_2d_x.values.shape != (nx,):
                raise ValueError(
                    f"2D x-axis shape {self.axis_2d_x.values.shape} != image width {nx}"
                )
            if self.axis_2d_y.values is not None and self.axis_2d_y.values.shape != (ny,):
                raise ValueError(
                    f"2D y-axis shape {self.axis_2d_y.values.shape} != image height {ny}"
                )
            if self.sigma_2d is not None and self.sigma_2d.shape != self.intensity_2d.shape:
                raise ValueError(
                    f"2D sigma shape {self.sigma_2d.shape} != "
                    f"intensity shape {self.intensity_2d.shape}"
                )

    @classmethod
    def from_results(
        cls,
        *,
        label: int | str,
        result_1d: Any | None = None,
        result_2d: Any | None = None,
        raw: np.ndarray | None = None,
        thumbnail: np.ndarray | None = None,
        mask_baked: bool = False,
        metadata_raw: Mapping[str, Any] | None = None,
        metadata_numeric: Mapping[str, float] | None = None,
        incident_angle: float | None = None,
        geometry: FrameGeometry | None = None,
        source_path: str | None = None,
        source_frame_index: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> "FrameView":
        axis_1d = intensity_1d = sigma_1d = None
        if result_1d is not None:
            unit = getattr(result_1d, "unit", "") or ""
            axis_1d = axis_from_unit(unit, getattr(result_1d, "radial", None))
            intensity_1d = getattr(result_1d, "intensity", None)
            sigma_1d = getattr(result_1d, "sigma", None)

        axis_2d_x = axis_2d_y = intensity_2d = sigma_2d = None
        two_d_kind = TwoDKind.Q_CHI
        if result_2d is not None:
            x_unit = getattr(result_2d, "unit", "") or ""
            y_unit = getattr(result_2d, "azimuthal_unit", "") or ""
            axis_2d_x = axis_from_unit(x_unit, getattr(result_2d, "radial", None))
            axis_2d_y = axis_from_unit(y_unit, getattr(result_2d, "azimuthal", None))
            raw_i2d = getattr(result_2d, "intensity", None)
            intensity_2d = None if raw_i2d is None else np.asarray(raw_i2d).T
            raw_s2d = getattr(result_2d, "sigma", None)
            sigma_2d = None if raw_s2d is None else np.asarray(raw_s2d).T
            two_d_kind = two_d_kind_from_units(x_unit, y_unit)

        return cls(
            label=label,
            axis_1d=axis_1d,
            intensity_1d=intensity_1d,
            sigma_1d=sigma_1d,
            axis_2d_x=axis_2d_x,
            axis_2d_y=axis_2d_y,
            intensity_2d=intensity_2d,
            sigma_2d=sigma_2d,
            two_d_kind=two_d_kind,
            raw=raw,
            thumbnail=thumbnail,
            mask_baked=mask_baked,
            metadata_raw=metadata_raw or {},
            metadata_numeric=metadata_numeric or numeric_metadata(metadata_raw or {}),
            incident_angle=incident_angle,
            geometry=geometry,
            source_path=source_path,
            source_frame_index=source_frame_index,
            extra=extra or {},
        )


_UNIT_LABELS = {
    "q_A^-1": "Q",
    "q_nm^-1": "Q",
    "1/angstrom": "Q",
    "angstrom^-1": "Q",
    "2th_deg": "2θ",
    "2th_rad": "2θ",
    "degrees": "χ",
    "deg": "χ",
    "chi_deg": "χ",
    "chi_rad": "χ",
    "qip_A^-1": "Q_ip",
    "qip_nm^-1": "Q_ip",
    "qoop_A^-1": "Q_oop",
    "qoop_nm^-1": "Q_oop",
    "qtot_A^-1": "Q_total",
    "qtot_nm^-1": "Q_total",
    "chigi_deg": "χ_GI",
    "chigi_rad": "χ_GI",
    "exit_angle_horz_deg": "exit angle horizontal",
    "exit_angle_vert_deg": "exit angle vertical",
}


def axis_from_unit(unit: str | None, values: Any | None = None, *, label: str | None = None) -> Axis:
    """Build an :class:`Axis` from a pyFAI/NeXus-style unit string."""

    unit = "" if unit is None else str(unit)
    return Axis(label=label or _UNIT_LABELS.get(unit, unit or "axis"), unit=unit, values=values)


def two_d_kind_from_units(x_unit: str | None, y_unit: str | None) -> TwoDKind:
    """Infer 2D axis identity from units when an explicit kind is absent."""

    x = (x_unit or "").lower()
    y = (y_unit or "").lower()
    if x.startswith("qip") and y.startswith("qoop"):
        return TwoDKind.QIP_QOOP
    if x.startswith("qtot") and y.startswith("chigi"):
        return TwoDKind.QTOT_CHIGI
    if x.startswith("exit_angle") or y.startswith("exit_angle"):
        return TwoDKind.EXIT_ANGLES
    return TwoDKind.Q_CHI


def numeric_metadata(metadata: Mapping[str, Any]) -> dict[str, float]:
    """Return scalar numeric values from a metadata mapping."""

    out: dict[str, float] = {}
    for key, value in metadata.items():
        try:
            arr = np.asarray(value)
            if arr.shape != ():
                continue
            val = float(arr)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            out[str(key)] = val
    return out


def _assert_array_close(name: str, a: np.ndarray | None, b: np.ndarray | None, *, rtol: float, atol: float) -> None:
    if a is None or b is None:
        if a is not None or b is not None:
            raise AssertionError(f"{name}: one side is None")
        return
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


def assert_frameview_equivalent(
    a: FrameView,
    b: FrameView,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Assert two frame views carry equivalent display/round-trip data."""

    if a.two_d_kind != b.two_d_kind:
        raise AssertionError(f"two_d_kind differs: {a.two_d_kind!r} != {b.two_d_kind!r}")
    for name in ("axis_1d", "axis_2d_x", "axis_2d_y"):
        ax_a = getattr(a, name)
        ax_b = getattr(b, name)
        if ax_a is None or ax_b is None:
            if ax_a is not None or ax_b is not None:
                raise AssertionError(f"{name}: one side is None")
            continue
        if ax_a.label != ax_b.label or ax_a.unit != ax_b.unit or ax_a.log != ax_b.log:
            raise AssertionError(f"{name} identity differs: {ax_a!r} != {ax_b!r}")
        _assert_array_close(f"{name}.values", ax_a.values, ax_b.values, rtol=rtol, atol=atol)
    for name in ("intensity_1d", "sigma_1d", "intensity_2d", "sigma_2d", "thumbnail"):
        _assert_array_close(name, getattr(a, name), getattr(b, name), rtol=rtol, atol=atol)
    if a.mask_baked != b.mask_baked:
        raise AssertionError(f"mask_baked differs: {a.mask_baked!r} != {b.mask_baked!r}")
    if a.incident_angle is None or b.incident_angle is None:
        if a.incident_angle is not None or b.incident_angle is not None:
            raise AssertionError("incident_angle: one side is None")
    elif not np.isclose(a.incident_angle, b.incident_angle, rtol=rtol, atol=atol, equal_nan=True):
        raise AssertionError(f"incident_angle differs: {a.incident_angle!r} != {b.incident_angle!r}")
