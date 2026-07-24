# -*- coding: utf-8 -*-
"""GI theta-motor default policy — compatibility re-export.

The ONE implementation now lives in :mod:`xrd_tools.session.gi_motor` (Qt-free,
so the headless run-configuration freeze and the GUI dropdowns share exactly one
policy — O-1a-ii item 5 / R4B-8).  This module keeps the historical import path
for the GUI consumers (integrator, image wrangler, Controls V2) working.
"""

from __future__ import annotations

from xrd_tools.session.gi_motor import (  # noqa: F401
    GI_MOTOR_PREFERENCE,
    pick_default_gi_motor,
)

__all__ = ["GI_MOTOR_PREFERENCE", "pick_default_gi_motor"]
