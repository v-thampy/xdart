"""Shared pytest setup for xdart.

Keep pyqtgraph on the same Qt binding as the generated UI modules before test
modules import ``pyqtgraph.Qt`` directly.
"""

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
