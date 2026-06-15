"""Canonical per-frame display/round-trip records.

These types are intentionally GUI-free.  They describe the data needed to
display, validate, stitch, fit, or round-trip one processed detector frame
without depending on xdart's live objects or Qt state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from xrd_tools.core.metadata import numeric_metadata


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
            # pyFAI result containers are (radial, azimuthal); FrameView's
            # display convention is (y, x), matching the saved stack reader.
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


def view_to_result_1d(view: "FrameView"):
    """Inverse of :meth:`FrameView.from_results`' 1D leg.

    Returns an :class:`IntegrationResult1D` reconstructed from the view's 1D
    axis + intensity (the round-trip the per-mode NeXus writer consumes), or
    ``None`` when the view carries no 1D data.
    """
    if (view.axis_1d is None or view.intensity_1d is None
            or view.axis_1d.values is None):
        return None
    from xrd_tools.core.containers import IntegrationResult1D

    return IntegrationResult1D(
        radial=np.asarray(view.axis_1d.values),
        intensity=np.asarray(view.intensity_1d),
        sigma=None if view.sigma_1d is None else np.asarray(view.sigma_1d),
        unit=view.axis_1d.unit or "",
    )


def view_to_result_2d(view: "FrameView"):
    """Inverse of :meth:`FrameView.from_results`' 2D leg.

    ``FrameView.intensity_2d`` is stored ``(ny, nx) = (axis_2d_y, axis_2d_x)``
    while ``IntegrationResult2D.intensity`` is ``(radial, azimuthal) =
    (axis_2d_x, axis_2d_y)``, so the intensity/sigma are transposed back (the
    exact inverse of ``from_results``' ``.T``).  ``None`` when no 2D data.
    """
    if (view.axis_2d_x is None or view.axis_2d_y is None or view.intensity_2d is None
            or view.axis_2d_x.values is None or view.axis_2d_y.values is None):
        return None
    from xrd_tools.core.containers import IntegrationResult2D

    return IntegrationResult2D(
        radial=np.asarray(view.axis_2d_x.values),
        azimuthal=np.asarray(view.axis_2d_y.values),
        intensity=np.asarray(view.intensity_2d).T,
        sigma=None if view.sigma_2d is None else np.asarray(view.sigma_2d).T,
        unit=view.axis_2d_x.unit or "",
        azimuthal_unit=view.axis_2d_y.unit or "",
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
    """Infer 2D axis identity from units when an explicit kind is absent.

    GI matching is deliberately lenient (substring) because existing
    processed files carry several unit spellings — canonical
    ``qip_A^-1``/``qoop_A^-1`` and ``exit_angle_horz_deg``/
    ``exit_angle_vert_deg``, but also legacy xdart ``horiz_exit``/
    ``vert_exit``.  Misclassifying a GI map as Q_CHI sends its qip axis
    through the q→2θ conversion downstream (arcsin out of range →
    collapsed/blank cake), so leniency is the safe direction.  This is
    the ONE classifier — xdart's display layer maps these kinds to its
    legacy strings at the display edge.
    """

    x = (x_unit or "").lower()
    y = (y_unit or "").lower()
    if "qip" in x or "qip" in y or "qoop" in x or "qoop" in y:
        return TwoDKind.QIP_QOOP
    if x.startswith("qtot") and y.startswith("chigi"):
        return TwoDKind.QTOT_CHIGI
    if "exit" in x or "exit" in y:
        return TwoDKind.EXIT_ANGLES
    return TwoDKind.Q_CHI


def _assert_array_close(name: str, a: np.ndarray | None, b: np.ndarray | None, *, rtol: float, atol: float) -> None:
    if a is None or b is None:
        if a is not None or b is not None:
            raise AssertionError(f"{name}: one side is None")
        return
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


def _assert_numeric_metadata_equivalent(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    rtol: float,
    atol: float,
) -> None:
    if set(a) != set(b):
        raise AssertionError(
            "metadata_numeric keys differ: "
            f"{sorted(a)!r} != {sorted(b)!r}"
        )
    for key in a:
        if not np.isclose(a[key], b[key], rtol=rtol, atol=atol, equal_nan=True):
            raise AssertionError(
                f"metadata_numeric[{key!r}] differs: {a[key]!r} != {b[key]!r}"
            )


def assert_frameview_equivalent(
    a: FrameView,
    b: FrameView,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Assert two frame views carry equivalent display/round-trip data."""

    if a.label != b.label:
        raise AssertionError(f"label differs: {a.label!r} != {b.label!r}")
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
    _assert_numeric_metadata_equivalent(
        a.metadata_numeric,
        b.metadata_numeric,
        rtol=rtol,
        atol=atol,
    )


# ---------------------------------------------------------------------------
# Multi-result records (ADR-0003 / ADR-0005)
# ---------------------------------------------------------------------------

DEFAULT_MODE_KEY = "default"
"""Reserved mode key for the active/top-level (non-GI single-result) slot.

A standard (non-GI) scan carries exactly one result per dimension under this
trivial key, which is also the slot persisted at the canonical top-level
``integrated_1d`` / ``integrated_2d`` NeXus group.  GI sub-modes use canonical
keys (``q_total``/``q_ip``/``q_oop``/``exit_angle`` for 1D, etc.); the
``mode_key -> on-disk subgroup name`` mapping will be established canonically in
:mod:`xrd_tools.io.schema` (Step 1) and is never derived from GUI label strings.
"""


def _view_1d_only(view: FrameView) -> FrameView:
    """Project ``view`` to a dimension-pure 1D view (2D fields dropped)."""
    return replace(
        view,
        axis_2d_x=None,
        axis_2d_y=None,
        intensity_2d=None,
        sigma_2d=None,
        two_d_kind=TwoDKind.Q_CHI,
    )


def _view_2d_only(view: FrameView) -> FrameView:
    """Project ``view`` to a dimension-pure 2D view (1D fields dropped)."""
    return replace(view, axis_1d=None, intensity_1d=None, sigma_1d=None)


def _merge_views(v1: FrameView | None, v2: FrameView | None) -> FrameView:
    """Merge a 1D-bearing and a 2D-bearing view into one display view.

    Shared per-frame fields (raw/thumbnail/metadata/geometry/source) are
    taken from the 1D view when present, falling back to the 2D view.  The two
    are the same frame, so they carry identical shared fields; the array fields
    use a None-check and the mapping fields use a presence (truthiness) check —
    "v1 wins when it has the field" — which only matters in the degenerate case
    where the two ever disagree (not expected).  The single-sided branches
    project to a dimension-pure view so the result never leaks the missing
    dimension even if an impure view is passed in directly.
    """
    if v1 is None:
        if v2 is None:
            raise ValueError("cannot project an empty FrameRecord")
        return _view_2d_only(v2)
    if v2 is None:
        return _view_1d_only(v1)
    return FrameView(
        label=v1.label,
        axis_1d=v1.axis_1d,
        intensity_1d=v1.intensity_1d,
        sigma_1d=v1.sigma_1d,
        axis_2d_x=v2.axis_2d_x,
        axis_2d_y=v2.axis_2d_y,
        intensity_2d=v2.intensity_2d,
        sigma_2d=v2.sigma_2d,
        two_d_kind=v2.two_d_kind,
        raw=v1.raw if v1.raw is not None else v2.raw,
        thumbnail=v1.thumbnail if v1.thumbnail is not None else v2.thumbnail,
        mask_baked=bool(v1.mask_baked or v2.mask_baked),
        metadata_raw=v1.metadata_raw if v1.metadata_raw else v2.metadata_raw,
        metadata_numeric=v1.metadata_numeric if v1.metadata_numeric else v2.metadata_numeric,
        incident_angle=(
            v1.incident_angle if v1.incident_angle is not None else v2.incident_angle
        ),
        geometry=v1.geometry if v1.geometry is not None else v2.geometry,
        source_path=v1.source_path if v1.source_path is not None else v2.source_path,
        source_frame_index=(
            v1.source_frame_index
            if v1.source_frame_index is not None
            else v2.source_frame_index
        ),
        extra=v1.extra if v1.extra else v2.extra,
    )


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """The multi-result record for one frame (ADR-0003).

    A reduction *completion* is single-result; a frame *record* is the union
    of every integration mode computed for that frame over its lifetime.  The
    1D and 2D sub-mode selections are independent (separate GUI combos), so
    results are kept in two per-dimension maps keyed by mode, each with its
    own active key.  Every stored value is a dimension-pure :class:`FrameView`
    (1D entries carry no 2D arrays and vice versa), so the equivalence atom
    :func:`assert_frameview_equivalent` applies per ``(frame, mode)`` unchanged.

    This type is intentionally GUI-free: the publication verdict / diagnostics
    are a thin xdart wrapper *over* a record, never fields of it.
    """

    label: int | str
    results_1d: Mapping[str, FrameView] = field(default_factory=dict)
    results_2d: Mapping[str, FrameView] = field(default_factory=dict)
    active_mode_1d: str = DEFAULT_MODE_KEY
    active_mode_2d: str = DEFAULT_MODE_KEY

    def __post_init__(self) -> None:
        for dim, results in (("results_1d", self.results_1d), ("results_2d", self.results_2d)):
            for mode, view in results.items():
                if not isinstance(view, FrameView):
                    raise TypeError(
                        f"{dim}[{mode!r}] must be a FrameView, got {type(view).__name__}"
                    )
        # Enforce dimension purity structurally: a 1D entry never carries 2D
        # arrays and vice versa, regardless of how the record was built (incl.
        # the direct-construction reload path).  _view_*_only is idempotent, so
        # already-pure entries (from from_view / with_result_*) cost one shallow
        # replace each.
        object.__setattr__(
            self,
            "results_1d",
            MappingProxyType({k: _view_1d_only(v) for k, v in self.results_1d.items()}),
        )
        object.__setattr__(
            self,
            "results_2d",
            MappingProxyType({k: _view_2d_only(v) for k, v in self.results_2d.items()}),
        )
        if self.results_1d and self.active_mode_1d not in self.results_1d:
            raise ValueError(
                f"active_mode_1d {self.active_mode_1d!r} not in "
                f"results_1d {sorted(self.results_1d)!r}"
            )
        if self.results_2d and self.active_mode_2d not in self.results_2d:
            raise ValueError(
                f"active_mode_2d {self.active_mode_2d!r} not in "
                f"results_2d {sorted(self.results_2d)!r}"
            )

    @property
    def is_empty(self) -> bool:
        return not self.results_1d and not self.results_2d

    @property
    def modes_1d(self) -> tuple[str, ...]:
        return tuple(self.results_1d)

    @property
    def modes_2d(self) -> tuple[str, ...]:
        return tuple(self.results_2d)

    def has_mode_1d(self, mode: str) -> bool:
        return mode in self.results_1d

    def has_mode_2d(self, mode: str) -> bool:
        return mode in self.results_2d

    def view_1d(self, mode: str | None = None) -> FrameView | None:
        return self.results_1d.get(mode if mode is not None else self.active_mode_1d)

    def view_2d(self, mode: str | None = None) -> FrameView | None:
        return self.results_2d.get(mode if mode is not None else self.active_mode_2d)

    def project(self, mode_1d: str | None = None, mode_2d: str | None = None) -> FrameView:
        """Merge the selected (default: active) 1D + 2D modes into one view."""
        return _merge_views(self.view_1d(mode_1d), self.view_2d(mode_2d))

    def active_view(self) -> FrameView:
        """The display view for the currently active 1D + 2D modes."""
        return self.project()

    def with_result_1d(
        self, mode: str, view: FrameView, *, make_active: bool = True
    ) -> "FrameRecord":
        """Return a new record with ``view`` upserted under 1D ``mode``."""
        results = dict(self.results_1d)
        results[mode] = view  # constructor enforces dimension purity
        return replace(
            self,
            results_1d=results,
            active_mode_1d=mode if make_active else self.active_mode_1d,
        )

    def with_result_2d(
        self, mode: str, view: FrameView, *, make_active: bool = True
    ) -> "FrameRecord":
        """Return a new record with ``view`` upserted under 2D ``mode``."""
        results = dict(self.results_2d)
        results[mode] = view  # constructor enforces dimension purity
        return replace(
            self,
            results_2d=results,
            active_mode_2d=mode if make_active else self.active_mode_2d,
        )

    @classmethod
    def from_view(
        cls,
        view: FrameView,
        *,
        mode_1d: str = DEFAULT_MODE_KEY,
        mode_2d: str = DEFAULT_MODE_KEY,
    ) -> "FrameRecord":
        """Build a single-mode record from one combined :class:`FrameView`.

        The bridge from today's single-result world: the view's 1D and 2D
        payloads become the sole entries of the per-dimension maps under the
        given keys (defaulting to :data:`DEFAULT_MODE_KEY`).
        """
        results_1d: dict[str, FrameView] = {}
        results_2d: dict[str, FrameView] = {}
        if view.has_1d:
            results_1d[mode_1d] = view  # constructor enforces dimension purity
        if view.has_2d:
            results_2d[mode_2d] = view  # constructor enforces dimension purity
        return cls(
            label=view.label,
            results_1d=results_1d,
            results_2d=results_2d,
            active_mode_1d=mode_1d if view.has_1d else DEFAULT_MODE_KEY,
            active_mode_2d=mode_2d if view.has_2d else DEFAULT_MODE_KEY,
        )


def assert_framerecord_equivalent(
    a: FrameRecord,
    b: FrameRecord,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Assert two records carry equivalent results for every mode.

    The multi-mode equivalence gate (ADR-0005): same label, same active
    keys, same mode sets, and each stored ``(frame, mode)`` view equivalent
    by :func:`assert_frameview_equivalent`.
    """
    if a.label != b.label:
        raise AssertionError(f"label differs: {a.label!r} != {b.label!r}")
    if a.active_mode_1d != b.active_mode_1d:
        raise AssertionError(
            f"active_mode_1d differs: {a.active_mode_1d!r} != {b.active_mode_1d!r}"
        )
    if a.active_mode_2d != b.active_mode_2d:
        raise AssertionError(
            f"active_mode_2d differs: {a.active_mode_2d!r} != {b.active_mode_2d!r}"
        )
    for dim, ra, rb in (
        ("results_1d", a.results_1d, b.results_1d),
        ("results_2d", a.results_2d, b.results_2d),
    ):
        if set(ra) != set(rb):
            raise AssertionError(
                f"{dim} mode sets differ: {sorted(ra)!r} != {sorted(rb)!r}"
            )
        for mode in ra:
            try:
                assert_frameview_equivalent(ra[mode], rb[mode], rtol=rtol, atol=atol)
            except AssertionError as exc:
                raise AssertionError(f"{dim}[{mode!r}]: {exc}") from exc
