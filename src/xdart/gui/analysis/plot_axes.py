# -*- coding: utf-8 -*-
"""The right-hand Y axis helpers, for the Scan Plot until it mounts the shared owners.

FIT-TOOLS-2026-10: the right-axis behaviour moved, once, into the shared trend owner
(``tabs/scattering/trend_view.py``: ``attach_right_axis``, ``add_right_series``, ``set_right_axis_visible``),
which every fit window mounts; this module only names them for the Scan Plot dialog, whose move onto the owners
is its own (phase 1).
"""

from xdart.gui.tabs.scattering.trend_view import add_right_series, attach_right_axis, set_right_axis_visible

__all__ = ["add_right_series", "attach_right_axis", "set_right_axis_visible"]
