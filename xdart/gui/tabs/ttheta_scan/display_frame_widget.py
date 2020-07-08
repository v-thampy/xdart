# -*- coding: utf-8 -*-
"""
@author: walroth
"""

# Standard library imports
import traceback

# Other imports
import numpy as np
from matplotlib import pyplot as plt

# Qt imports
import pyqtgraph as pg
from pyqtgraph import Qt
from pyqtgraph.Qt import QtWidgets

PUSHBUTTON_STYLESHEET = """
            QPushButton {
              background-color: #505F69;
              border: 1px solid #32414B;
              color: #F0F0F0;
              border-radius: 4px;
              padding: 3px;
              outline: none;
              min-width: 20px;
            }
            QPushButton:disabled {
              background-color: #32414B;
              border: 1px solid #32414B;
              color: #787878;
              border-radius: 4px;
              padding: 3px;
            }
            
            QPushButton:checked {
              background-color: #32414B;
              border: 1px solid #32414B;
              border-radius: 4px;
              padding: 3px;
              outline: none;
            }
            
            QPushButton:checked:disabled {
              background-color: #19232D;
              border: 1px solid #32414B;
              color: #787878;
              border-radius: 4px;
              padding: 3px;
              outline: none;
            }
            
            QPushButton:checked:selected {
              background: #1464A0;
              color: #32414B;
            }
            
            QPushButton::menu-indicator {
              subcontrol-origin: padding;
              subcontrol-position: bottom right;
              bottom: 4px;
            }
            
            QPushButton:pressed {
              background-color: #19232D;
              border: 1px solid #19232D;
            }
            
            QPushButton:pressed:hover {
              border: 1px solid #148CD2;
            }
            
            QPushButton:hover {
              border: 1px solid #148CD2;
              color: #F0F0F0;
            }
            
            QPushButton:selected {
              background: #1464A0;
              color: #32414B;
            }
            
            QPushButton:hover {
              border: 1px solid #148CD2;
              color: #F0F0F0;
            }
            
            QPushButton:focus {
              border: 1px solid #1464A0;
            }
        """
QFileDialog = QtWidgets.QFileDialog

# This module imports
from .displayFrameUI import Ui_Form
from ...gui_utils import RectViewBox, get_rect
from ...widgets import XDImageWidget
import xdart.utils as ut

formats = [
    str(f.data(), encoding='utf-8').lower() for f in
    Qt.QtGui.QImageReader.supportedImageFormats()
]

class displayFrameWidget(Qt.QtWidgets.QWidget):
    """Widget for displaying 2D image data and 1D plots from EwaldSphere
    objects. 
    
    attributes:
        auto_last: bool, whether to automatically select latest arch
        curve1: pyqtgraph pen, overall data line
        curve2: pyqtgraph pen, individual arch data line
        histogram: pyqtgraph HistogramLUTWidget, used for adjusting min
            and max level for image
        image: pyqtgraph ImageItem, displays the 2D data
        image_plot: pyqtgraph plot, for 2D data
        image_win: pyqtgraph GraphicsLayoutWidget, layout for the 2D
            data
        imageViewBox: RectViewBox, used to set behavior of the image
            plot
        plot: pyqtgraph plot, for 1D data
        plot_layout: QVBoxLayout, for holding the 1D plotting widgets
        plot_win: pyqtgraph GraphicsLayoutWidget, layout for the 1D
            data
        sphere: EwaldSphere, unused.
        ui: Ui_Form from qtdeisgner
    
    methods:
        get_arch_data_2d: Gets 2D data from an arch object
        get_sphere_data_2d: Gets overall 2D data for the sphere
        update: Updates the displayed image and plot
        update_image: Updates image data based on selections
        update_plot: Updates plot data based on selections
    """
    def __init__(self, sphere, arch, parent=None):
        _translate = Qt.QtCore.QCoreApplication.translate
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.imageUnit.setItemText(0, _translate("Form", "2" + u"\u03B8"))
        self.ui.plotUnit.setItemText(0, _translate("Form", "2" + u"\u03B8"))

        # Data object initialization
        self.sphere = sphere
        self.arch = arch

        # State variable initialization
        self.auto_last = False

        # Image pane setup
        self.image_layout = Qt.QtWidgets.QHBoxLayout(self.ui.imageFrame)
        self.image_layout.setContentsMargins(0, 0, 0, 0)
        self.image_layout.setSpacing(0)
        self.image_widget = XDImageWidget()
        self.image_layout.addWidget(self.image_widget)

        # Image pane signal connections
        self.ui.imageMethod.setCurrentIndex(1)
        self.ui.imageMethod.setEnabled(False)

        self.plot_layout = Qt.QtWidgets.QVBoxLayout(self.ui.plotFrame)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_win = pg.GraphicsLayoutWidget()
        self.plot_layout.addWidget(self.plot_win)
        vb = RectViewBox()
        self.plot = self.plot_win.addPlot(viewBox=vb)
        self.curve1 = self.plot.plot(pen=(50,100,255))
        self.curve2 = self.plot.plot(
            pen=(200,50,50,200), 
            symbolBrush=(200,50,50,200), 
            symbolPen=(0,0,0,0), 
            symbolSize=4
        )

        self.ui.plotMethod.setCurrentIndex(1)
        self.ui.plotMethod.setEnabled(False)
        
        # Signal connections
        self.ui.imageIntRaw.activated.connect(self.update_image)
        self.ui.imageMethod.activated.connect(self.update_image)
        self.ui.imageUnit.activated.connect(self.update_image)
        self.ui.imageNRP.activated.connect(self.update_image)
        self.ui.imageMask.stateChanged.connect(self.update_image)
        self.ui.shareAxis.stateChanged.connect(self.update)
        self.ui.plotMethod.activated.connect(self.update_plot)
        self.ui.plotUnit.activated.connect(self.update_plot)
        self.ui.plotNRP.activated.connect(self.update_plot)
        self.ui.plotOverlay.stateChanged.connect(self.update_plot)

        # Fix width of buttons
        self.ui.pushRight.setStyleSheet(PUSHBUTTON_STYLESHEET)
        self.ui.pushRightLast.setStyleSheet(PUSHBUTTON_STYLESHEET)
        self.ui.pushLeft.setStyleSheet(PUSHBUTTON_STYLESHEET)
        self.ui.pushLeftLast.setStyleSheet(PUSHBUTTON_STYLESHEET)

        #self.update()

    def update(self):
        """Updates image and plot frames based on toolbar options
        """
        # Sets title text
        if self.arch.idx is None:
            self.ui.labelCurrent.setText(self.sphere.name)
        else:
            self.ui.labelCurrent.setText("Image " + str(self.arch.idx))

        if self.ui.shareAxis.isChecked():
            self.ui.plotUnit.setCurrentIndex(self.ui.imageUnit.currentIndex())
            self.ui.plotUnit.setEnabled(False)
            self.plot.setXLink(self.image_widget.image_plot)
        
        else:
            self.plot.setXLink(None)
            self.ui.plotUnit.setEnabled(True)
        
        try:
            self.update_image()
        except TypeError:
            return False
        try:
            self.update_plot()
        except TypeError:
            return False
        return True
    
    def update_image(self):
        """Updates image plotted in image frame
        """
        if self.sphere.name == 'null_main':
            data = np.arange(100).reshape(10,10)
            rect = Qt.QtCore.QRect(1,1,1,1)
        else:
            try:
                if self.arch.idx is not None:
                    data, rect = self.get_arch_data_2d()

                else:
                    data, rect = self.get_sphere_data_2d()
            except (TypeError, IndexError):
                data = np.arange(100).reshape(10, 10)
                rect = Qt.QtCore.QRect(1, 1, 1, 1)
        
        self.image_widget.setImage(data)
        self.image_widget.setRect(rect)
        
        return data

    def get_arch_data_2d(self):
        """Returns data and QRect for data in arch
        """
        with self.arch.arch_lock:
            int_data = self.arch.int_2d
        
        if self.ui.imageIntRaw.currentIndex() == 0:
            data, corners, sigma = read_NRP(self.ui.imageNRP, int_data)
            
            rect = get_rect(
                get_xdata(self.ui.imageUnit, int_data)[corners[2]:corners[3]], 
                int_data.chi[corners[0]:corners[1]]
            )
        
        elif self.ui.imageIntRaw.currentIndex() == 1:
            with self.arch.arch_lock:
                if self.ui.imageNRP.currentIndex() == 0:
                    if self.arch.map_norm is None or self.arch.map_norm == 0:
                        data = self.arch.map_raw.copy()
                    else:
                        data = self.arch.map_raw.copy()/self.arch.map_norm
                else:
                    data = self.arch.map_raw.copy()
                if self.ui.imageMask.isChecked():
                    data.ravel()[self.arch.mask] = data.max()
            rect = get_rect(
                np.arange(data.shape[0]), 
                np.arange(data.shape[1]),
            )
        
        return data, rect

    def get_sphere_data_2d(self):
        """Returns data and QRect for data in sphere
        """
        with self.sphere.sphere_lock:
            if self.ui.imageMethod.currentIndex() == 0:
                int_data = self.sphere.mgi_2d
                if type(int_data.ttheta) == int:
                    self.ui.imageMethod.setCurrentIndex(1)
                    int_data = self.sphere.bai_2d
            elif self.ui.imageMethod.currentIndex() == 1:
                int_data = self.sphere.bai_2d
        
        data, corners, sigma = read_NRP(self.ui.imageNRP, int_data)
        
        rect = get_rect(
            get_xdata(self.ui.imageUnit, int_data)[corners[2]:corners[3]], 
            int_data.chi[corners[0]:corners[1]]
        )
        
        return data, rect
    
    def update_plot(self):
        """Updates data in plot frame
        """
        if self.sphere.name == 'null_main':
            data = (np.arange(100), np.arange(100))
            self.curve1.setData(data[0], data[1])
            self.curve2.setData(data[0], data[1])
            return data
        
        else:
            try:
                with self.sphere.sphere_lock:
                    if self.ui.plotMethod.currentIndex() == 0:
                        sphere_int_data = self.sphere.mgi_1d
                        if type(sphere_int_data.ttheta) == int:
                            self.ui.plotMethod.setCurrentIndex(1)
                            sphere_int_data = self.sphere.bai_1d
                    elif self.ui.plotMethod.currentIndex() == 1:
                        sphere_int_data = self.sphere.bai_1d

                s_ydata, corners, s_sigma = read_NRP(self.ui.plotNRP, sphere_int_data)
                s_xdata = get_xdata(self.ui.plotUnit, sphere_int_data)[corners[0]:corners[1]]

                if self.arch.idx is not None:
                    with self.arch.arch_lock:
                        arc_int_data = self.arch.int_1d

                    if self.ui.plotOverlay.isChecked():
                        self.curve1.setData(s_xdata, s_ydata)
                    else:
                        self.curve1.clear()

                    a_ydata, corners, a_sigma = read_NRP(self.ui.plotNRP, arc_int_data)
                    a_xdata = get_xdata(self.ui.plotUnit, arc_int_data)[corners[0]:corners[1]]
                    self.curve2.setData(a_xdata, a_ydata)

                    return a_xdata, a_ydata, a_sigma

                else:
                    self.curve1.setData(s_xdata, s_ydata)
                    self.curve2.clear()

                    return s_xdata, s_ydata, s_sigma

            except (TypeError, IndexError):
                data = (np.arange(100), np.arange(100))
                self.curve1.setData(data[0], data[1])
                self.curve2.setData(data[0], data[1])
                return data

    def save_image(self):
        """Saves currently displayed image. Formats are automatically
        grabbed from Qt. Also implements tiff saving.
        """
        ext_filter = "Images ("
        for f in formats:
            ext_filter += "*." + f + " "

        ext_filter += "*.tiff)"

        fname, _ = QFileDialog.getSaveFileName(filter=ext_filter)
        if fname == '':
            return

        _, ext = fname.split('.')
        if ext.lower() in formats:
            self.image_widget.imageItem.save(fname)
        
        elif ext.lower() == 'tiff':
            data = self.update_image()
            plt.imsave(fname, data.T, cmap='gray')
    
    def save_array(self):
        """Saves currently displayed data. Currently supports .xye
        and .csv.
        """
        fname, _ = QFileDialog.getSaveFileName(
            filter="XRD Files (*.xye *.csv)"
        )
        if fname == '':
            return

        xdata, ydata, sigma = self.update_plot()

        _, ext = fname.split('.')
        if ext.lower() == 'xye':
            ut.write_xye(fname, xdata, ydata, sigma)
        
        elif ext.lower() == 'csv':
            ut.write_csv(fname, xdata, ydata, sigma)


def read_NRP(box, int_data):
    """Reads the norm, raw, pcount option box and returns
    appropriate ydata.
    
    args:
        box: QComboBox, list of choices for picking data to return
        int_data: int_nd_data object, data to parse
    
    returns:
        data: numpy array, non-zero region from nzarray based on
            choices in box
        corners: tuple, the bounds of the non-zero region of the
            dataset
    """
    if box.currentIndex() == 0:
        nzarr = int_data.norm
        sigmanz = int_data.sigma
    elif box.currentIndex() == 1:
        nzarr = int_data.raw
        sigmanz = int_data.sigma_raw
    elif box.currentIndex() == 2:
        nzarr = int_data.pcount
        sigmanz = None
    data = nzarr.data[()].T
    sigma = sigmanz.data[()].T
    corners = nzarr.corners
    
    if data.size == 0:
        data = np.zeros(int_data.norm.shape)
        sigma = np.zeros_like(data)
        if len(corners) == 2:
            corners = [0, int_data.norm.shape[0]]
        elif len(corners) == 4:
            corners = [0, int_data.norm.shape[0], 0, int_data.norm.shape[1]]
    return data, corners, sigma


def get_xdata(box, int_data):
    """Reads the unit box and returns appropriate xdata
    
    args:
        box: QComboBox, list of options
        int_data: int_nd_data object, data to parse
    
    returns:
        xdata: numpy array, x axis data for plot.
    """
    if box.currentIndex() == 0:
        xdata = int_data.ttheta
    elif box.currentIndex() == 1:
        xdata = int_data.q
    return xdata


