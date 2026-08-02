"""Scientific axis compatibility and trace projection for the vNext shell."""

from __future__ import annotations

import os

import numpy as np

from xrd_tools.core import (
    TwoDKind,
    axis_from_unit,
    convert_radial_axis,
)
from xrd_tools.session.display_logic import (
    canonical_axis_key,
    nanmean_slice,
)

from .display_values import (
    DisplayFrameKey,
    StandardDisplayPayload,
    display_payload_is_valid,
)
from .shell_values import (
    AxisProjection,
    FrameNavigationProjection,
    HeavyProjection,
    TraceProjection,
)


_PLOT_AXIS_KEYS = {
    "Q": "q_A^-1",
    "2theta": "2th_deg",
    "chi": "chi_deg",
    "q_ip": "qip_A^-1",
    "q_oop": "qoop_A^-1",
    "exit_angle": "exit_angle_deg",
    "chi_gi": "chigi_deg",
}
_PLOT_AXIS_CHOICES = {
    "q_A^-1": "Q",
    "2th_deg": "2theta",
    "chi_deg": "chi",
    "qip_A^-1": "q_ip",
    "qoop_A^-1": "q_oop",
    "exit_angle_deg": "exit_angle",
    "chigi_deg": "chi_gi",
}
_CONVERTIBLE_RADIAL_AXIS_KEYS = frozenset({"q_A^-1", "2th_deg"})
_SHARE_PLOT_AXIS_BY_IMAGE = {
    "Q-Chi": "Q",
    "2Th-Chi": "2theta",
    "qip_qoop": "q_ip",
    "q_chi": "Q",
    "exit_angles": "exit_angle",
}
_LEGEND_AXIS_LABELS = {
    "q_A^-1": "Q",
    "2th_deg": "2θ",
    "chi_deg": "χ",
    "qip_A^-1": "Qᵢₚ",
    "qoop_A^-1": "Qₒₒₚ",
    "exit_angle_deg": "exit",
    "chigi_deg": "χGI",
}


def share_plot_axis_for_image(image_axis: str) -> str | None:
    """Return the plot selector that shares the rendered cake x-axis."""

    return _SHARE_PLOT_AXIS_BY_IMAGE.get(str(image_axis))


def payload_is_qualified(
    payload: object,
    navigation: FrameNavigationProjection,
) -> bool:
    if type(payload) is not StandardDisplayPayload:
        return False
    frame = payload.frame_key
    if (
        type(frame) is not DisplayFrameKey
        or not any(candidate is frame for candidate in navigation.frames)
        or not (
            frame is navigation.current
            or any(
                candidate is frame
                for candidate in navigation.selected
            )
        )
    ):
        return False
    try:
        return display_payload_is_valid(
            payload,
            frame.run_identity,
            frame,
            payload.selection_generation,
        )
    except Exception:
        return False


def trace_projection(
    payload: StandardDisplayPayload,
    *,
    requested_axis: str,
    allow_cake: bool,
    slice_enabled: bool,
    slice_center: float,
    slice_width: float,
) -> TraceProjection | None:
    try:
        frame = payload.frame_key
        view = payload.view
        if type(frame) is not DisplayFrameKey:
            return None
        native = _native_trace_values(payload, requested_axis)
        cake_matches, cake, slice_axis = (
            _cake_trace_values(
                payload,
                requested_axis=requested_axis,
                slice_enabled=slice_enabled,
                slice_center=slice_center,
                slice_width=slice_width,
            )
            if allow_cake
            else (False, None, None)
        )
        requested_key = _PLOT_AXIS_KEYS.get(
            requested_axis,
            requested_axis,
        )
        native_exact = (
            native is not None
            and _axis_key(native[0]) == requested_key
        )
        cake_exact = (
            cake is not None
            and _axis_key(cake[0]) == requested_key
        )
        # A native/cake axis uses the accepted 1-D result until the user
        # enables a complementary-axis slice. A cake-only axis is projected
        # from the full cake even with slicing disabled.
        selected_from_cake = False
        if cake_matches and slice_enabled:
            selected = cake
            selected_from_cake = True
        elif cake is not None and (
            native is None or cake_exact and not native_exact
        ):
            selected = cake
            selected_from_cake = True
        else:
            selected = native or cake
        if selected is None:
            return None
        axis, intensity = selected
        # Match the production accumulator presentation: scan name plus the
        # one-based frame number. The persisted frame key remains unchanged.
        source_name = os.path.basename(str(view.source_path or ""))
        source_suffix = os.path.splitext(source_name)[1].casefold()
        frame_label = (
            int(view.source_frame_index) + 1
            if view.source_frame_index is not None
            and source_suffix in {".h5", ".hdf5", ".nxs"}
            else frame.local_frame_label
        )
        title = (
            f"{frame.source_scan}_{frame_label}"
            if frame.source_scan and frame.source_scan != "null_main"
            else str(frame_label)
        )
        if (
            selected_from_cake
            and slice_enabled
            and slice_axis is not None
        ):
            title += (
                f" · {_legend_axis_label(axis)}@"
                f"{_legend_axis_label(slice_axis)}="
                f"{float(slice_center):.2f}±"
                f"{float(slice_width):.2f}"
            )
        return TraceProjection(
            frame,
            axis,
            intensity,
            title,
            _finite_metadata_value(view.metadata_numeric, "epoch"),
        )
    except Exception:
        return None


def _native_trace_values(
    payload: StandardDisplayPayload,
    requested_axis: str,
) -> tuple[AxisProjection, np.ndarray] | None:
    view = payload.view
    axis = view.axis_1d
    intensity = view.intensity_1d
    if (
        axis is None
        or axis.values is None
        or intensity is None
        or axis.values.shape != intensity.shape
    ):
        return None
    native = AxisProjection(axis.values, axis.label, axis.unit)
    requested_key = _PLOT_AXIS_KEYS.get(requested_axis, requested_axis)
    native_key = _axis_key(native)
    if (
        native_key == requested_key
        or {
            native_key,
            requested_key,
        } <= _CONVERTIBLE_RADIAL_AXIS_KEYS
    ):
        native = _present_radial_axis(
            native,
            requested_axis=requested_axis,
            wavelength_m=payload.wavelength_m,
        )
    return (
        native,
        intensity,
    )


def _cake_trace_values(
    payload: StandardDisplayPayload,
    *,
    requested_axis: str,
    slice_enabled: bool,
    slice_center: float,
    slice_width: float,
) -> tuple[
    bool,
    tuple[AxisProjection, np.ndarray] | None,
    AxisProjection | None,
]:
    """Project one requested 1-D axis from ``(cake_y, cake_x)`` values."""

    view = payload.view
    intensity = view.intensity_2d
    x_axis = view.axis_2d_x
    y_axis = view.axis_2d_y
    if (
        intensity is None
        or intensity.ndim != 2
        or x_axis is None
        or y_axis is None
        or x_axis.values is None
        or y_axis.values is None
        or x_axis.values.shape != (intensity.shape[1],)
        or y_axis.values.shape != (intensity.shape[0],)
    ):
        return False, None, None
    requested_key = _PLOT_AXIS_KEYS.get(requested_axis, requested_axis)
    x_projection = AxisProjection(
        x_axis.values,
        x_axis.label,
        x_axis.unit,
    )
    y_projection = AxisProjection(
        y_axis.values,
        y_axis.label,
        y_axis.unit,
    )
    x_key = _axis_key(x_projection)
    y_key = _axis_key(y_projection)
    x_matches = (
        x_key == requested_key
        or {
            x_key,
            requested_key,
        } <= _CONVERTIBLE_RADIAL_AXIS_KEYS
    )
    if x_matches:
        selected_axis = _present_radial_axis(
            x_projection,
            requested_axis=requested_axis,
            wavelength_m=payload.wavelength_m,
        )
        slice_axis = y_projection
        if slice_enabled:
            selected = (
                slice_center - slice_width <= y_projection.values
            ) & (
                y_projection.values <= slice_center + slice_width
            )
            reduced = nanmean_slice(intensity[selected, :], 0)
        else:
            reduced = nanmean_slice(intensity, 0)
    elif y_key == requested_key:
        selected_axis = y_projection
        slice_axis = x_projection
        if slice_enabled:
            selected = (
                slice_center - slice_width <= x_projection.values
            ) & (
                x_projection.values <= slice_center + slice_width
            )
            reduced = nanmean_slice(intensity[:, selected], 1)
        else:
            reduced = nanmean_slice(intensity, 1)
    else:
        return False, None, None
    if (
        reduced is None
        or reduced.ndim != 1
        or reduced.shape != selected_axis.values.shape
    ):
        return True, None, slice_axis
    return True, (selected_axis, reduced), slice_axis


def _legend_axis_label(axis: AxisProjection) -> str:
    return _LEGEND_AXIS_LABELS.get(
        _axis_key(axis),
        str(axis.label),
    )


def _axis_key(axis: AxisProjection) -> str:
    return canonical_axis_key(f"{axis.label} ({axis.unit})")


def _finite_metadata_value(metadata, key: str) -> float | None:
    try:
        value = float(metadata[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def heavy_projection(
    payload: StandardDisplayPayload,
    *,
    requested_axis: str = "Q-Chi",
) -> HeavyProjection | None:
    try:
        frame = payload.frame_key
        if type(frame) is not DisplayFrameKey:
            return None
        view = payload.view
        raw = view.raw if view.raw is not None else view.thumbnail
        cake = view.intensity_2d
        cake_x = view.axis_2d_x
        cake_y = view.axis_2d_y
        x_axis = (
            None
            if cake_x is None or cake_x.values is None
            else AxisProjection(
                cake_x.values,
                cake_x.label,
                cake_x.unit,
            )
        )
        y_axis = (
            None
            if cake_y is None or cake_y.values is None
            else AxisProjection(
                cake_y.values,
                cake_y.label,
                cake_y.unit,
            )
        )
        if x_axis is not None and view.two_d_kind is TwoDKind.Q_CHI:
            x_axis = _present_radial_axis(
                x_axis,
                requested_axis=requested_axis,
                wavelength_m=payload.wavelength_m,
            )
        complete_cake = (
            cake is not None
            and x_axis is not None
            and y_axis is not None
            and x_axis.values.shape == (cake.shape[1],)
            and y_axis.values.shape == (cake.shape[0],)
        )
        if not complete_cake:
            cake = None
            x_axis = None
            y_axis = None
        if raw is None and cake is None:
            return None
        return HeavyProjection(frame, raw, cake, x_axis, y_axis)
    except Exception:
        return None


def _present_radial_axis(
    axis: AxisProjection,
    *,
    requested_axis: str,
    wavelength_m: float | None,
) -> AxisProjection:
    target = {
        "Q-Chi": "q_A^-1",
        "2Th-Chi": "2th_deg",
        "Q": "q_A^-1",
        "2theta": "2th_deg",
    }.get(requested_axis)
    if target is None:
        return axis
    wavelength_A = (
        None
        if wavelength_m is None
        else float(wavelength_m) * 1.0e10
    )
    try:
        values = convert_radial_axis(
            axis.values,
            axis.unit,
            target,
            wavelength_A,
        )
    except (TypeError, ValueError, FloatingPointError):
        return axis
    presented = axis_from_unit(target, values)
    return AxisProjection(
        presented.values,
        presented.label,
        presented.unit,
    )


def requested_image_axis(
    payload: StandardDisplayPayload,
    preference: str,
) -> str:
    """GI cakes are native mode results, not Q/2theta display conversions."""

    if payload.measurement_mode != "GI":
        return preference
    return {
        "q_chi": "Q-Chi",
    }.get(payload.gi_mode_2d, payload.gi_mode_2d)


def image_axis_choice(
    axis: AxisProjection | None,
    kind: TwoDKind,
    *,
    measurement_mode: str,
    gi_mode_2d: str,
) -> str:
    if measurement_mode == "GI":
        if gi_mode_2d in {"qip_qoop", "q_chi", "exit_angles"}:
            return gi_mode_2d
        return {
            TwoDKind.QIP_QOOP: "qip_qoop",
            TwoDKind.EXIT_ANGLES: "exit_angles",
        }.get(kind, "q_chi")
    if kind is not TwoDKind.Q_CHI:
        return "Qz-Qxy"
    if axis is None:
        return "Q-Chi"
    key = canonical_axis_key(f"{axis.label} ({axis.unit})")
    return "2Th-Chi" if key == "2th_deg" else "Q-Chi"


def plot_axis_choice(
    traces: tuple[TraceProjection, ...],
    fallback: str,
) -> str:
    keys = {
        canonical_axis_key(f"{trace.axis.label} ({trace.axis.unit})")
        for trace in traces
    }
    return (
        _PLOT_AXIS_CHOICES.get(next(iter(keys)), fallback)
        if len(keys) == 1
        else fallback
    )


__all__ = [
    "heavy_projection",
    "image_axis_choice",
    "payload_is_qualified",
    "plot_axis_choice",
    "requested_image_axis",
    "share_plot_axis_for_image",
    "trace_projection",
]
