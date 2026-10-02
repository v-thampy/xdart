"""Compatibility name for the shared, Qt-free RSM operator values."""
import sys
from xrd_tools.analysis import rsm_values as _owner

sys.modules[__name__] = _owner
