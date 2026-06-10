# -*- coding: utf-8 -*-
"""
@author: walroth, vthampy
"""
# Top level script for running gui based program

# Standard library imports
import sys
import gc
import os
import signal
import logging
import faulthandler
faulthandler.enable()  # Print Python traceback on bus error / segfault

# Set PySide6 as the Qt binding for pyqtgraph before any Qt imports.
# Also export MPLBACKEND so child processes (e.g. pyFAI-calib2) inherit it.
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'
os.environ['MPLBACKEND'] = 'QtAgg'

# Set matplotlib backend before any matplotlib import can occur.
# Use QtAgg (the Qt6 backend) to match pyqtgraph's choice.
import matplotlib
matplotlib.use('QtAgg')

# Default root logging level — INFO is what every wrangler log line
# currently uses, so basicConfig(INFO) is enough to surface them.
# The DEBUG line below opts specific loggers into more verbose output;
# the basicConfig must happen first so the handler's threshold is open.
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
)

# Suppress pyFAI INFO logs (e.g. "No sensor configuration provided").
logging.getLogger('pyFAI').setLevel(logging.WARNING)
logging.getLogger('pyFAI.gui.matplotlib').setLevel(logging.ERROR)
# Suppress silx's "pyOpenCL has been imported but can't be used here"
# warning — OpenCL is optional and the message has no user action.
logging.getLogger('silx.opencl').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# C4: minimum compatible ssrl_xrd_tools version.  MUST equal the
# ``ssrl_xrd_tools>=`` floor in pyproject.toml (tests/test_min_ssrl_version.py
# asserts they match).  The pip floor only protects pip installs — the
# documented dev workflow is an editable install from a sibling clone, which
# bypasses it entirely; this runtime guard turns "crashes on the first write"
# into a clear startup error.
MIN_SSRL_VERSION = "0.41.0"


def _version_tuple(v):
    """Lenient (major, minor, patch) for X.Y.Z-style strings."""
    parts = []
    for tok in str(v).split(".")[:3]:
        digits = ""
        for ch in tok:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits or 0))
    return tuple(parts + [0] * (3 - len(parts)))


def _ssrl_capabilities_ok():
    """Probe the load-bearing symbols xdart hard-requires.  Used to tolerate a
    STALE editable-install version stamp (metadata only refreshes on
    ``pip install -e``, not on ``git pull``) when the code is actually new
    enough."""
    try:
        import inspect
        from ssrl_xrd_tools.io.read import relative_source_path  # noqa: F401
        from ssrl_xrd_tools.reduction import ReductionSession
        return ("join_timeout" in inspect.signature(
                    ReductionSession.finish).parameters
                and hasattr(ReductionSession, "drain")
                # 0.41.0 symbols — without these the probe approves a checkout
                # that crashes at every session open (open_live_reduction_session
                # passes retain_products= unguarded).  Keep this list in sync
                # with the NEWEST load-bearing ssrl APIs xdart calls.
                and "retain_products" in inspect.signature(
                    ReductionSession).parameters
                and hasattr(ReductionSession, "release_products"))
    except Exception:
        return False


def check_ssrl_version():
    """Fail loudly at startup on an incompatible ssrl_xrd_tools install."""
    try:
        import ssrl_xrd_tools
        have = getattr(ssrl_xrd_tools, "__version__", "0.0.0")
    except ImportError as exc:
        raise SystemExit(
            f"xdart requires ssrl_xrd_tools>={MIN_SSRL_VERSION} "
            f"(import failed: {exc})")
    if _version_tuple(have) >= _version_tuple(MIN_SSRL_VERSION):
        return
    if _ssrl_capabilities_ok():
        # The code has everything we need; only the metadata stamp is old
        # (editable install not re-installed since the version bump).
        logger.warning(
            "ssrl_xrd_tools reports %s (< required %s) but provides all "
            "required APIs — the editable install's version stamp is likely "
            "stale; re-run 'pip install -e ../ssrl_xrd_tools' to refresh it.",
            have, MIN_SSRL_VERSION)
        return
    raise SystemExit(
        f"xdart requires ssrl_xrd_tools>={MIN_SSRL_VERSION}, found {have}. "
        f"Editable installs bypass the pip floor — update and reinstall the "
        f"sibling ssrl_xrd_tools checkout.")

# Qt imports
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    QtGui: Any = None
    QtWidgets: Any = None
else:
    from pyqtgraph.Qt import QtGui, QtWidgets

# This module imports
from xdart.gui.mainWindow import Ui_MainWindow
from xdart.gui import tabs


QMainWindow = QtWidgets.QMainWindow


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('xdart')
        self.ui.actionOpen.triggered.connect(self.openFile)
        self.ui.actionExit.triggered.connect(self.exit)
        self.fname = None

        # Embed the main widget directly (no tab container).
        # The widget chooses its own scratch directory via get_fname_dir().
        self.main_widget = tabs.static_scan.staticWidget()
        self.setCentralWidget(self.main_widget)

        self.show()
        # Size to the screen (was a fixed 1600x920): at 1600 wide the right
        # panel's ~418px minimum content width overshoots its 24% share and
        # clamps the middle display panels well below their intended 57%.
        try:
            avail = self.screen().availableGeometry()
            self.resize(int(avail.width() * 0.92), int(avail.height() * 0.92))
        except Exception:
            self.resize(1600, 920)

    def exit(self):
        try:
            self.main_widget.close()
        finally:
            self.close()
            gc.collect()
            try:
                os.killpg(os.getpid(), signal.SIGTERM)
            except ProcessLookupError:
                pass
            sys.exit(0)

    def openFile(self):
        try:
            self.main_widget.open_file()
        except Exception:
            logger.exception("Error opening file")


def main():
    check_ssrl_version()
    app = QtWidgets.QApplication(sys.argv)
    # N8: apply dark theme before any widget construction so
    # pyqtgraph plot backgrounds are set in time (pyqtgraph
    # snapshots the config at widget creation).
    try:
        from xdart.gui.themes import apply_dark_theme
        apply_dark_theme(app)
    except Exception:
        logger.exception("Failed to apply dark theme; using Qt default")
    mw = Main()
    mw.show()
    app.exec()


if __name__ == '__main__':
    sys.exit(main())
