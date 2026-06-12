"""Shared pytest setup for xdart.

Keep pyqtgraph on the same Qt binding as the generated UI modules before test
modules import ``pyqtgraph.Qt`` directly.
"""

import os
import tempfile

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_API", "PySide6")
os.environ.setdefault("MPLBACKEND", "Agg")

# Isolate session persistence: staticWidget restores ~/.xdart/session.json at
# construction and SAVES it at close() -- without this, GUI-test fixtures
# polluted the user's real session (and inherited the user's state, making
# tests order/machine dependent).
os.environ.setdefault(
    "XDART_SESSION_FILE",
    os.path.join(tempfile.mkdtemp(prefix="xdart_test_session_"),
                 "session.json"),
)
