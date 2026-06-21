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

# pyqtgraph's log-axis tick painter computes 10**range while the histogram
# axis still holds the previous LINEAR image's extent for one paint after a
# Log toggle (e.g. Eiger counts ~4e9 -> 10**4e9).  Harmless — the inf is
# clamped on the next paint — but it logged a RuntimeWarning on every
# toggle.  Scoped to exactly that message and module.
import warnings
warnings.filterwarnings(
    'ignore', message='overflow encountered in power',
    category=RuntimeWarning, module=r'pyqtgraph\.graphicsItems\.AxisItem')

logger = logging.getLogger(__name__)

# Qt imports
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    QtCore: Any = None
    QtGui: Any = None
    QtWidgets: Any = None
else:
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

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
        self._init_theme_menu()
        self.fname = None

        # Embed the main widget directly (no tab container).
        # The widget chooses its own scratch directory via get_fname_dir().
        self.main_widget = tabs.static_scan.staticWidget()
        self.setCentralWidget(self.main_widget)
        # D2 (greenfield Phase 3): in the live app, rehydrate evicted frames off
        # the GUI thread so scroll-back no longer freezes on a ~5 s .nxs open.
        # Done here (not in widget construction) so headless widget tests keep
        # their synchronous reads.
        try:
            self.main_widget.enable_async_hydration()
        except Exception:
            pass

        # Default size: 90% of the available screen, centered (was a fixed
        # 1600x920, whose width clamped the middle display panels below
        # their intended 57% share).  setGeometry rather than resize() --
        # a post-show resize was unreliable for width on macOS.
        self.show()
        try:
            avail = self.screen().availableGeometry()
            w = int(avail.width() * 0.95)
            h = int(avail.height() * 0.90)
            self.setGeometry(avail.x() + (avail.width() - w) // 2,
                             avail.y() + (avail.height() - h) // 2, w, h)
        except Exception:
            self.resize(1600, 920)

    def _init_theme_menu(self):
        """Add a Config ▸ Theme (Dark/Light) toggle, persisted in QSettings.

        Built in code (not the .ui) so the menu exists without regenerating the
        designer file.  Switching re-applies the QSS live; the pyqtgraph plot
        canvas background is snapshotted at widget creation, so a full plot
        recolor still needs a relaunch (noted for a later stage)."""
        settings = QtCore.QSettings("xdart", "xdart")
        current = settings.value("theme", "dark")
        if current not in ("dark", "light"):
            current = "dark"
        menubar = self.menuBar()
        config_menu = None
        for menu in menubar.findChildren(QtWidgets.QMenu):
            if menu.title().replace("&", "").strip().lower() == "config":
                config_menu = menu
                break
        if config_menu is None:
            config_menu = menubar.addMenu("Config")
        theme_menu = config_menu.addMenu("Theme")
        group = QtGui.QActionGroup(self)
        group.setExclusive(True)
        for name, label in (("dark", "Dark"), ("light", "Light")):
            action = QtGui.QAction(label, self)
            action.setCheckable(True)
            action.setChecked(name == current)
            action.triggered.connect(
                lambda _checked=False, n=name: self._set_theme(n))
            group.addAction(action)
            theme_menu.addAction(action)

    def _set_theme(self, name):
        """Apply theme ``name`` live and persist the choice."""
        from xdart.gui.themes import apply_theme
        app = QtWidgets.QApplication.instance()
        if app is not None:
            apply_theme(app, name)
        QtCore.QSettings("xdart", "xdart").setValue("theme", name)

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


def _apply_cli_session_args(argv):
    """Parse ``-f``/``-n`` and point the session system at the right file via
    env vars BEFORE any widget loads its session.  Returns the argv (minus the
    consumed flags) to hand to Qt.

    ``xdart -f``      → fresh session (load nothing, persist nothing).
    ``xdart -n NAME`` → named saved session (NAME under ~/.xdart; the ``.json``
                        extension is forced if the user omits it).
    """
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(
        prog='xdart', description='xdart — SSRL XRD reduction GUI')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-f', '--fresh', action='store_true',
        help='start a fresh session (does not load or modify your saved session)')
    group.add_argument(
        '-n', '--session', metavar='NAME',
        help='start from a named saved session (NAME under ~/.xdart; '
             '.json is appended if omitted)')
    args, rest = parser.parse_known_args(argv[1:])
    if args.fresh:
        os.environ['XDART_SESSION_FRESH'] = '1'
    elif args.session:
        name = args.session
        if not name.lower().endswith('.json'):
            name += '.json'                  # force the .json extension
        p = Path(name)
        if not p.is_absolute() and p.parent == Path('.'):
            p = Path.home() / '.xdart' / name   # bare name -> ~/.xdart/
        os.environ['XDART_SESSION_FILE'] = str(p)
    return [argv[0], *rest]


def run():
    argv = _apply_cli_session_args(sys.argv)
    app = QtWidgets.QApplication(argv)
    # N8: apply the saved theme before any widget construction so
    # pyqtgraph plot backgrounds are set in time (pyqtgraph
    # snapshots the config at widget creation).
    try:
        from xdart.gui.themes import apply_theme
        theme = QtCore.QSettings("xdart", "xdart").value("theme", "dark")
        if theme not in ("dark", "light"):
            theme = "dark"
        apply_theme(app, theme)
    except Exception:
        logger.exception("Failed to apply theme; using Qt default")
    mw = Main()
    mw.show()
    app.exec()


main = run   # back-compat alias


if __name__ == '__main__':
    sys.exit(run())
