"""Compatibility name for native Stitch preparation's shared headless owner."""
import sys
from xrd_tools.analysis import stitch_native_values as _owner

sys.modules[__name__] = _owner
