"""Deprecated compatibility shim: ``ssrl_xrd_tools`` is now ``xrd_tools``.

The package was renamed in the xrd-tools monorepo (2026-06).  This shim
keeps old user code/notebooks importable:

    import ssrl_xrd_tools            -> xrd_tools
    from ssrl_xrd_tools.io import …  -> xrd_tools.io

First-party code must NEVER import this module (guarded by a test).
"""
import sys
import warnings

import xrd_tools as _xrd_tools

warnings.warn(
    "ssrl_xrd_tools has been renamed to xrd_tools (xrd-tools monorepo); "
    "update imports — this shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Alias the package and every already/late-imported submodule: rebinding
# sys.modules entries makes ``from ssrl_xrd_tools.io.read import X`` resolve
# through the real package machinery.
sys.modules[__name__] = _xrd_tools
