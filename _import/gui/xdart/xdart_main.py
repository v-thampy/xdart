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

# Suppress pyFAI INFO logs (e.g. "No sensor configuration provided").
logging.getLogger('pyFAI').setLevel(logging.WARNING)
logging.getLogger('pyFAI.gui.matplotlib').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

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


def setup_data_folders(exp_list):
    """
    Creates xdart/data folder and xdart/data/tabs folder for storing
    local data. These are ignored by gitignore.

    Parameters
    ----------
    exp_list : list, set of tabs to be

    Returns
    -------
    tab_paths : dict, paths for tabs to store data
    """
    current_directory = os.path.dirname(__file__)
    data_directory = os.path.join(current_directory, "../data")
    if not os.path.isdir(data_directory):
        os.mkdir(data_directory)
    tab_paths = {}
    for e in exp_list:
        tab_directory = os.path.join(data_directory, e)
        tab_paths[e] = tab_directory
        if not os.path.isdir(tab_directory):
            os.mkdir(tab_directory)
    return tab_paths


QMainWindow = QtWidgets.QMainWindow


class Main(QMainWindow):
    def __init__(self, tab_paths):
        """
        Parameters
        ----------
        tab_paths : dict
        """
        super().__init__()
        self.tab_paths = tab_paths
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('xdart')
        self.ui.actionOpen.triggered.connect(self.openFile)
        self.ui.actionExit.triggered.connect(self.exit)
        self.fname = None

        # Embed the main widget directly (no tab container)
        self.main_widget = tabs.static_scan.staticWidget(
            local_path=self.tab_paths['static_scan'])
        self.setCentralWidget(self.main_widget)

        self.show()
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
    tab_paths = setup_data_folders(tabs.exp_list)
    app = QtWidgets.QApplication(sys.argv)
    mw = Main(tab_paths)
    mw.show()
    app.exec()


if __name__ == '__main__':
    sys.exit(main())
