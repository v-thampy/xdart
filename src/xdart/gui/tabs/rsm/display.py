"""The committed RSM's six products as Int's scientific view draws them (AUD-9 stage 2b G7).

The page builds a ``KeyedPanelsProjection`` from the headless ``RSMViewerModel`` snapshot, the projection
authority (its slices and finite mean projections unchanged), and ``ScientificView.reconcile_keyed`` draws
it: the viewer model's 2×3 layout (``rsm_viewer_layout``: the three slices through the selected indices,
then the three mean projections) on Int's image panes and 1-D plots, under Int's own top bar (colour map,
Log). Each slice spans its bins' outer edges (``rsm_axis_bounds``), so each pixel sits at its bin centre
(Codex's stage-1 review F3). No widget is built here (hard rule 14).
"""
from __future__ import annotations

import numpy as np

from xdart.gui.tabs.scattering.shell_values import (
    AxisProjection, KeyedImageProjection, KeyedPanelsProjection, KeyedTraceProjection,
)
from xrd_tools.session.display_logic import PanelRole
from xrd_tools.session.rsm_viewer_model import rsm_axis_bounds, rsm_viewer_layout

#: each panel's (horizontal, vertical) axis indices, in the layout's order: the slices' pairs, then each
#: projection's axis. A slice's normal is the axis it leaves out.
_AXES = ((0, 1), (0, 2), (1, 2), (0, None), (1, None), (2, None))


def slice_normal(horizontal: int, vertical: int) -> int:
    """The axis a slice through ``horizontal`` and ``vertical`` is taken along."""
    return 3 - horizontal - vertical


def rsm_slice_normals(values) -> dict[str, int]:
    """Each slice panel's instance, by the axis its index steps along (0 H, 1 K, 2 L; or Qx, Qy, Qz)."""
    keys = [key for row in rsm_viewer_layout(values.coordinate_frame) for key in row]
    return {key.instance: slice_normal(*axes) for key, axes in zip(keys, _AXES, strict=True)
            if key.role is PanelRole.SLICE_2D}


def rsm_keyed_projection(values, snapshot, *, color_map: str = "Default", log_scale: bool = False,
                         title: str = "", background_enabled: bool = False,
                         background_set: bool = False) -> KeyedPanelsProjection:
    """The six products of ``snapshot`` (``values``' viewer-model snapshot) as a keyed layout."""
    labels = values.coordinate_frame.axis_labels
    symbols = values.coordinate_frame.axis_symbols
    axes = tuple(np.asarray(axis) for _name, axis in values.axes)
    indices = snapshot.state.axis_indices
    layout = rsm_viewer_layout(values.coordinate_frame)
    keys = tuple(key for row in layout for key in row)
    panels = []
    for key, product, (horizontal, vertical) in zip(keys, snapshot.products, _AXES, strict=True):
        if product.panel_key != key:
            raise ValueError("an RSM snapshot's products follow its layout")
        x_axis = AxisProjection(np.asarray(product.x_axis), labels[horizontal], "",
                                edges=rsm_axis_bounds(product.x_axis))
        if key.role is PanelRole.SLICE_2D:
            normal = slice_normal(horizontal, vertical)
            at = indices[normal]
            panels.append(KeyedImageProjection(
                key,
                f"{symbols[horizontal]}{symbols[vertical]} slice · {symbols[normal]} = "
                f"{float(axes[normal][at]):.6g} (index {at})",
                np.asarray(product.values),
                x_axis,
                AxisProjection(np.asarray(product.y_axis_or_none), labels[vertical], "",
                               edges=rsm_axis_bounds(product.y_axis_or_none)),
                at,
                tuple(str(index) for index in range(values.shape[normal])),
            ))
        else:
            panels.append(KeyedTraceProjection(
                key, f"{symbols[horizontal]} mean projection",
                AxisProjection(np.asarray(product.x_axis), labels[horizontal], ""),
                np.asarray(product.values), "Mean I"))
    status = (f"{values.coordinate_frame.display_name} · shape {values.shape} · "
              + " · ".join(f"{symbol} {axis[0]:.6g}..{axis[-1]:.6g}"
                           for symbol, axis in zip(symbols, axes, strict=True)))
    return KeyedPanelsProjection(layout, tuple(panels), title=title, status=status, color_map=color_map,
                                 log_scale=bool(log_scale), background_enabled=background_enabled,
                                 background_set=background_set)


__all__ = ["rsm_keyed_projection", "rsm_slice_normals", "slice_normal"]
