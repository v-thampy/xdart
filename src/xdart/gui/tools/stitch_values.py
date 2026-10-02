"""Compatibility name for the shared, Qt-free Stitch operator values."""
import sys
from xrd_tools.analysis import stitch_values as _owner

sys.modules[__name__] = _owner
