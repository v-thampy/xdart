"""The view preferences one plot command changes, by Int's rules: the reducer Int's workspace and the
Stitch page share (each adds its own side effects: selection, presentation, residency)."""
from __future__ import annotations

from dataclasses import replace

from .scientific_axes import share_plot_axis_for_image, slice_pins_for_plot_axis
from .scientific_plot_options import plot_option_update
from .shell_values import ShellCommand, ShellCommandKind

#: The 2-D maps the pane can show, and the plot modes.
IMAGE_AXES = frozenset({"Q-Chi", "2Th-Chi", "qip_qoop", "q_chi", "q_chi_derived", "exit_angles"})
PLOT_MODES = frozenset({"Single", "Overlay", "Average", "Sum", "Waterfall"})
_RANGES = {("q", "low"): ("q_range", 0), ("q", "high"): ("q_range", 1),
           ("chi", "low"): ("chi_range", 0), ("chi", "high"): ("chi_range", 1)}


def scientific_preference_updates(preferences, command: ShellCommand, *, shown_image_axis: str | None = None):
    """The fields of ``preferences`` (``ScientificPreferences``) that ``command`` changes, or None for a
    command this does not reduce or a value it refuses. A pin lives in Overlay and Waterfall only; a plot
    axis carries its pins; Share axis follows the map shown (``shown_image_axis``, else the preference)."""
    kind, value = command.kind, command.value

    def sharing(axis):  # the plot axis a shared image axis implies, with the pins it keeps
        return dict(plot_axis=axis, slice_pins=slice_pins_for_plot_axis(
            preferences.slice_pins, preferences.plot_axis, axis))
    if kind is ShellCommandKind.SET_NORM_CHANNEL:
        return {"norm_channel": str(value)}
    if kind is ShellCommandKind.SET_COLOR_MAP:
        return {"color_map": str(value)}
    if kind is ShellCommandKind.SET_LOG_SCALE:  # the image and the curve
        return {"log_scale": bool(value),
                "plot_options": replace(preferences.plot_options, intensity_scale="Log" if value else "Linear")}
    if kind is ShellCommandKind.SET_IMAGE_AXIS:
        if type(value) is not str or value not in IMAGE_AXES:
            return None
        # a display preference only: choosing which 2-D map the pane shows edits no run intent
        updates = {"image_axis": value}
        if preferences.share_axis:
            matching = share_plot_axis_for_image(value)
            updates.update({"share_axis": False} if matching is None else sharing(matching))
        return updates
    if kind is ShellCommandKind.SET_PLOT_AXIS:
        return sharing(str(value))
    if kind is ShellCommandKind.SET_PLOT_MODE:
        if type(value) is not str or value not in PLOT_MODES:
            return None
        return {"plot_mode": value, **({} if value in {"Overlay", "Waterfall"} else {"slice_pins": ()})}
    if kind is ShellCommandKind.SET_SHARE_AXIS:
        if type(value) is not bool:
            return None
        matching = share_plot_axis_for_image(preferences.image_axis if shown_image_axis is None else shown_image_axis)
        updates = {"share_axis": bool(value and matching is not None)}
        if value and matching is not None:
            updates.update(sharing(matching))
        return updates
    if kind is ShellCommandKind.SET_SLICE_ENABLED:
        return {"slice_enabled": bool(value)}
    if kind is ShellCommandKind.SET_SLICE_CENTER:
        return {"slice_center": float(value)}
    if kind is ShellCommandKind.SET_SLICE_WIDTH:
        return {"slice_width": float(value)}
    if kind is ShellCommandKind.SET_RANGE:
        edge = _RANGES.get(tuple(command.path))
        if type(value) not in {int, float} or edge is None:
            return None
        name, index = edge
        bounds = list(getattr(preferences, name))
        bounds[index] = float(value)
        return {name: tuple(bounds)}
    if kind is ShellCommandKind.SET_PLOT_OPTION:
        options = plot_option_update(preferences.plot_options, command.path, value)
        return None if options is None else {"plot_options": options}
    return None
