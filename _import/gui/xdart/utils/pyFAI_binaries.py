"""Wrappers around pyFAI GUI tools (calib2, drawmask).

Heavy imports (pyFAI.gui, silx.gui) are deferred to function bodies so they
don't run at application startup.  This avoids silx/Qt backend conflicts and
speeds up the initial load.
"""
import os
import logging

import pyFAI.resources
import pyFAI.calibrant
import pyFAI.detectors
import pyFAI.io.image
import fabio

from pyqtgraph import Qt

logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)
try:
    import hdf5plugin  # noqa
except ImportError:
    logger.debug("Unable to load hdf5plugin, backtrace:", exc_info=True)

QFileDialog = Qt.QtWidgets.QFileDialog
logger_uncaught = logging.getLogger("pyFAI-calib2.UNCAUGHT")


def pyFAI_calib2_main():
    # Lazy imports — only needed when user opens calibration window
    import silx
    from silx.gui import qt
    from pyFAI.gui.CalibrationWindow import CalibrationWindow
    from pyFAI.gui.CalibrationContext import CalibrationContext
    from pyFAI.app.calib2 import parse_options, setup_model

    # --help must also work without Qt
    options = parse_options()

    if options.debug:
        logging.root.setLevel(logging.DEBUG)

    if options.opengl:
        silx.config.DEFAULT_PLOT_BACKEND = "opengl"

    pyFAI.resources.silx_integration()
    settings = qt.QSettings(qt.QSettings.IniFormat,
                            qt.QSettings.UserScope,
                            "pyfai",
                            "pyfai-calib2",
                            None)

    context = CalibrationContext(settings)
    context.restoreSettings()

    setup_model(context.getCalibrationModel(), options)

    # Inline subclass to avoid top-level import of CalibrationWindow
    class CalibrationWindowXdart(CalibrationWindow):
        def __init__(self, ctx):
            super().__init__(context=ctx)
            self.context = ctx

        def closeEvent(self, event):
            poniFile = self.model().experimentSettingsModel().poniFile()
            if not poniFile.isSynchronized():
                button = qt.QMessageBox.question(
                    self,
                    "calib2",
                    "The PONI file was not saved.\nDo you really want to close the application?",
                    qt.QMessageBox.Cancel | qt.QMessageBox.No | qt.QMessageBox.Yes,
                    qt.QMessageBox.Yes)
                if button != qt.QMessageBox.Yes:
                    event.ignore()
                    return
            event.accept()
            CalibrationContext._releaseSingleton()

    window = CalibrationWindowXdart(context)
    window.setVisible(True)
    window.setAttribute(qt.Qt.WA_DeleteOnClose, True)
    context.saveSettings()


def pyFAI_drawmask_main(window, image, processFile):
    usage = "pyFAI-drawmask file1.edf file2.edf ..."
    description = """
    Draw a mask, i.e. an image containing the list of pixels which are considered invalid
    (no scintillator, module gap, beam stop shadow, ...).
    """

    window.setImageData(image)
    outfile = os.path.splitext(processFile)[0] + "-mask.edf"
    window.setOutputFile(outfile)
    window.outFile = outfile
    print("Your mask-file will be saved into %s" % outfile)


def get_MaskImageWidgetXdart():
    """Factory function — returns the MaskImageWidgetXdart class.

    Call this lazily instead of importing MaskImageWidgetXdart at the top level.
    """
    from pyFAI.app.drawmask import MaskImageWidget

    class MaskImageWidgetXdart(MaskImageWidget):
        """Window application which allows creating a mask manually."""

        def __init__(self):
            super().__init__()
            self.outFile = None

        def closeEvent(self, event):
            if os.path.exists(self.outFile):
                mask_file = os.path.basename(self.outFile)
                out_dialog = Qt.QtWidgets.QMessageBox()
                out_dialog.setText(f'{mask_file} saved in Image directory')
                out_dialog.exec()
            event.accept()

    return MaskImageWidgetXdart
