# -*- coding: utf-8 -*-
"""
Display constants — axis labels, units, and helper functions shared
across display_frame_widget and related modules.

Extracted from display_frame_widget.py to keep configuration separate
from widget logic.

@author: thampy
"""

import numpy as np
from .integrator import GI_MODES_1D, GI_MODES_2D  # noqa: F401

# ── Unicode symbols ──
AA_inv = u'\u212B\u207B\u00B9'
Th = u'\u03B8'
Chi = u'\u03C7'
Deg = u'\u00B0'
Qip_s = u'Q\u1D62\u209A'        # Q_ip subscript
Qoop_s = u'Q\u2092\u2092\u209A'  # Q_oop subscript
Qtot_s = u'Q\u209C\u2092\u209C'  # Q_tot subscript

# ── Standard (non-GI) axis options ──
plotUnits = [f"Q ({AA_inv})", f"2{Th} ({Deg})", f"{Chi} ({Deg})"]
imageUnits = [f"Q-{Chi}", f"2{Th}-{Chi}"]

x_labels_1D = ('Q', f"2{Th}", Chi)
x_units_1D = (AA_inv, Deg, Deg)

x_labels_2D = ('Q', f"2{Th}")
x_units_2D = (AA_inv, Deg)

y_labels_2D = (Chi, Chi)
y_units_2D = (Deg, Deg)

# ── GI axis options (order matches GI_MODES_1D / GI_MODES_2D) ──
gi_plotUnits = [
    f"Q ({AA_inv})",
    f"{Qip_s} ({AA_inv})",
    f"{Qoop_s} ({AA_inv})",
    f"Exit angle ({Deg})",
]
gi_imageUnits = [
    f"{Qip_s}-{Qoop_s}",
    f"Q-{Chi}",
    f"Exit angles",
]

gi_x_labels_1D = ('Q', Qip_s, Qoop_s, 'Exit angle')
gi_x_units_1D = (AA_inv, AA_inv, AA_inv, Deg)

gi_x_labels_2D = (Qip_s, 'Q', 'Exit angle')
gi_x_units_2D = (AA_inv, AA_inv, Deg)

gi_y_labels_2D = (Qoop_s, Chi, 'Exit angle')
gi_y_units_2D = (AA_inv, Deg, Deg)

# ── Axis decomposition for 2D → 1D slicing ──
# Maps each GI 2D mode to the pair of 1D axis labels available from it.
# The keys are GI_MODES_2D entries; values are (radial_label, azimuthal_label)
# using the same gi_plotUnits label strings for consistency.
GI_2D_AXES = {
    'qip_qoop': (
        f"{Qip_s} ({AA_inv})",   # radial  (x axis of 2D)
        f"{Qoop_s} ({AA_inv})",  # azimuthal (y axis of 2D)
    ),
    'q_chi': (
        f"Q ({AA_inv})",         # radial
        f"{Chi} ({Deg})",        # azimuthal
    ),
    'exit_angles': (
        f"Exit angle ({Deg})",   # both axes are exit angles
        f"Exit angle ({Deg})",
    ),
}

# Standard (non-GI) 2D axes
STD_2D_AXES = {
    0: (f"Q ({AA_inv})", f"{Chi} ({Deg})"),       # Q-Chi image
    1: (f"2{Th} ({Deg})", f"{Chi} ({Deg})"),       # 2Th-Chi image
}


def _downsample_for_display(data, widget):
    """Reduce array resolution to match widget pixel size. Display-only."""
    if data is None or data.ndim != 2:
        return data
    w = max(widget.width(), 200)
    h = max(widget.height(), 200)
    if data.shape[0] > w * 2 or data.shape[1] > h * 2:
        sx = max(1, data.shape[0] // w)
        sy = max(1, data.shape[1] // h)
        return data[::sx, ::sy]
    return data
