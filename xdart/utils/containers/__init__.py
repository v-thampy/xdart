"""
xdart container types.

New code should use the ssrl_xrd_tools types directly:

    from ssrl_xrd_tools.core.containers import PONI, IntegrationResult1D, IntegrationResult2D

The aliases below are kept for backward compatibility while the rest of
xdart is being migrated.  They will be removed once all callers have been
updated.
"""
from ssrl_xrd_tools.core.containers import (
    PONI,
    IntegrationResult1D,
    IntegrationResult2D,
)

from .compat import read_legacy_1d, read_legacy_2d, read_legacy_2d_gi

# ---------------------------------------------------------------------------
# Legacy name aliases – import these only if you haven't migrated yet.
# They are intentionally NOT exported via __all__ to discourage new use.
# ---------------------------------------------------------------------------
try:
    from .int_data_static import int_1d_data_static, int_2d_data_static
except Exception:  # pragma: no cover
    int_1d_data_static = None  # type: ignore[assignment,misc]
    int_2d_data_static = None  # type: ignore[assignment,misc]

try:
    from .poni import get_poni_dict
except Exception:  # pragma: no cover
    get_poni_dict = None  # type: ignore[assignment]

__all__ = [
    # New canonical types
    "PONI",
    "IntegrationResult1D",
    "IntegrationResult2D",
    # Compat readers
    "read_legacy_1d",
    "read_legacy_2d",
    "read_legacy_2d_gi",
]
