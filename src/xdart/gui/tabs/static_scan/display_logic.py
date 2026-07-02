# -*- coding: utf-8 -*-
"""Re-export shim for the static-scan display decision core.

The pure display-state/read-policy logic lives in
``xrd_tools.session.display_logic``.  Keep this module thin so existing xdart
imports continue to work while headless callers can use the xrd_tools home.
"""

from __future__ import annotations

from xrd_tools.session import display_logic as _display_logic
from xrd_tools.session.display_logic import *  # noqa: F401,F403 - compatibility shim

globals().update(
    {
        name: value
        for name, value in vars(_display_logic).items()
        if not name.startswith("__")
    }
)
__all__ = getattr(
    _display_logic,
    "__all__",
    tuple(name for name in vars(_display_logic) if not name.startswith("_")),
)
