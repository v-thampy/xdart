"""Back-compat re-export of :class:`DiffractometerConfig`.

The canonical definition lives in :mod:`ssrl_xrd_tools.core.geometry`
so that :mod:`ssrl_xrd_tools.core.config` can use it without forcing a
``core → rsm`` upward import. Existing callers (including notebooks and
user scripts) can continue to do::

    from ssrl_xrd_tools.rsm.geometry import DiffractometerConfig

unchanged.
"""
from __future__ import annotations

from ssrl_xrd_tools.core.geometry import DiffractometerConfig

__all__ = ["DiffractometerConfig"]
