# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'integratorUI_mac.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1440, 360)
        Form.setMaximumSize(QtCore.QSize(16777215, 360))
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame1D = QtWidgets.QFrame(Form)
        self.frame1D.setMaximumSize(QtCore.QSize(16777215, 190))
        self.frame1D.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame1D.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame1D.setLineWidth(20)
        self.frame1D.setMidLineWidth(5)
        self.frame1D.setObjectName("frame1D")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame1D)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame1D_layout = QtWidgets.QVBoxLayout()
        self.frame1D_layout.setSpacing(0)
        self.frame1D_layout.setObjectName("frame1D_layout")
        self.frame1D_header = QtWidgets.QFrame(self.frame1D)
        self.frame1D_header.setMaximumSize(QtCore.QSize(16777215, 50))
        self.frame1D_header.setAutoFillBackground(True)
        self.frame1D_header.setStyleSheet("")
        self.frame1D_header.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame1D_header.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame1D_header.setObjectName("frame1D_header")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame1D_header)
        self.horizontalLayout_4.setContentsMargins(15, 3, 5, 3)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label1D = QtWidgets.QLabel(self.frame1D_header)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label1D.sizePolicy().hasHeightForWidth())
        self.label1D.setSizePolicy(sizePolicy)
        self.label1D.setMinimumSize(QtCore.QSize(70, 0))
        self.label1D.setMaximumSize(QtCore.QSize(70, 16777215))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label1D.setFont(font)
        self.label1D.setAutoFillBackground(False)
        self.label1D.setStyleSheet("background-color: rgba(195, 195, 195, 200);")
        self.label1D.setAlignment(QtCore.Qt.AlignCenter)
        self.label1D.setObjectName("label1D")
        self.horizontalLayout_3.addWidget(self.label1D)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.axis1D = QtWidgets.QComboBox(self.frame1D_header)
        self.axis1D.setObjectName("axis1D")
        self.axis1D.addItem("")
        self.horizontalLayout_3.addWidget(self.axis1D)
        spacerItem1 = QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.label_npts_1D = QtWidgets.QLabel(self.frame1D_header)
        self.label_npts_1D.setMaximumSize(QtCore.QSize(40, 16777215))
        self.label_npts_1D.setIndent(-1)
        self.label_npts_1D.setObjectName("label_npts_1D")
        self.horizontalLayout_3.addWidget(self.label_npts_1D)
        self.npts_1D = QtWidgets.QLineEdit(self.frame1D_header)
        self.npts_1D.setMaximumSize(QtCore.QSize(55, 16777215))
        self.npts_1D.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.npts_1D.setObjectName("npts_1D")
        self.horizontalLayout_3.addWidget(self.npts_1D)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_3)
        self.frame1D_layout.addWidget(self.frame1D_header)
        self.frame1D_range = QtWidgets.QFrame(self.frame1D)
        self.frame1D_range.setMaximumSize(QtCore.QSize(16777215, 71))
        self.frame1D_range.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame1D_range.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame1D_range.setObjectName("frame1D_range")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame1D_range)
        self.horizontalLayout_2.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.gridLayout_1D = QtWidgets.QGridLayout()
        self.gridLayout_1D.setHorizontalSpacing(6)
        self.gridLayout_1D.setObjectName("gridLayout_1D")
        self.radial_autoRange_1D = QtWidgets.QCheckBox(self.frame1D_range)
        self.radial_autoRange_1D.setChecked(True)
        self.radial_autoRange_1D.setObjectName("radial_autoRange_1D")
        self.gridLayout_1D.addWidget(self.radial_autoRange_1D, 0, 4, 1, 1)
        self.unit_1D = QtWidgets.QComboBox(self.frame1D_range)
        self.unit_1D.setMaximumSize(QtCore.QSize(90, 16777215))
        self.unit_1D.setObjectName("unit_1D")
        self.unit_1D.addItem("")
        self.unit_1D.addItem("")
        self.gridLayout_1D.addWidget(self.unit_1D, 0, 0, 1, 1)
        self.label_to1 = QtWidgets.QLabel(self.frame1D_range)
        self.label_to1.setMaximumSize(QtCore.QSize(100, 30))
        self.label_to1.setObjectName("label_to1")
        self.gridLayout_1D.addWidget(self.label_to1, 1, 2, 1, 1)
        self.azim_low_1D = QtWidgets.QLineEdit(self.frame1D_range)
        self.azim_low_1D.setObjectName("azim_low_1D")
        self.gridLayout_1D.addWidget(self.azim_low_1D, 1, 1, 1, 1)
        self.radial_low_1D = QtWidgets.QLineEdit(self.frame1D_range)
        self.radial_low_1D.setObjectName("radial_low_1D")
        self.gridLayout_1D.addWidget(self.radial_low_1D, 0, 1, 1, 1)
        self.azim_autoRange_1D = QtWidgets.QCheckBox(self.frame1D_range)
        self.azim_autoRange_1D.setChecked(True)
        self.azim_autoRange_1D.setObjectName("azim_autoRange_1D")
        self.gridLayout_1D.addWidget(self.azim_autoRange_1D, 1, 4, 1, 1)
        self.azim_high_1D = QtWidgets.QLineEdit(self.frame1D_range)
        self.azim_high_1D.setObjectName("azim_high_1D")
        self.gridLayout_1D.addWidget(self.azim_high_1D, 1, 3, 1, 1)
        self.label_to2 = QtWidgets.QLabel(self.frame1D_range)
        self.label_to2.setObjectName("label_to2")
        self.gridLayout_1D.addWidget(self.label_to2, 0, 2, 1, 1)
        self.label_azim_1D = QtWidgets.QLabel(self.frame1D_range)
        self.label_azim_1D.setMinimumSize(QtCore.QSize(55, 0))
        self.label_azim_1D.setMaximumSize(QtCore.QSize(100, 50))
        self.label_azim_1D.setAlignment(QtCore.Qt.AlignCenter)
        self.label_azim_1D.setObjectName("label_azim_1D")
        self.gridLayout_1D.addWidget(self.label_azim_1D, 1, 0, 1, 1)
        self.radial_high_1D = QtWidgets.QLineEdit(self.frame1D_range)
        self.radial_high_1D.setObjectName("radial_high_1D")
        self.gridLayout_1D.addWidget(self.radial_high_1D, 0, 3, 1, 1)
        self.horizontalLayout_2.addLayout(self.gridLayout_1D)
        self.frame1D_layout.addWidget(self.frame1D_range)
        self.frame1D_buttons = QtWidgets.QFrame(self.frame1D)
        self.frame1D_buttons.setMaximumSize(QtCore.QSize(16777215, 71))
        self.frame1D_buttons.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame1D_buttons.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame1D_buttons.setObjectName("frame1D_buttons")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.frame1D_buttons)
        self.horizontalLayout_6.setContentsMargins(5, 5, 5, 20)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setSpacing(6)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.integrate1D = QtWidgets.QPushButton(self.frame1D_buttons)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.integrate1D.sizePolicy().hasHeightForWidth())
        self.integrate1D.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(13)
        self.integrate1D.setFont(font)
        self.integrate1D.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.integrate1D.setObjectName("integrate1D")
        self.horizontalLayout_5.addWidget(self.integrate1D)
        self.advanced1D = QtWidgets.QPushButton(self.frame1D_buttons)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.advanced1D.sizePolicy().hasHeightForWidth())
        self.advanced1D.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(13)
        self.advanced1D.setFont(font)
        self.advanced1D.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.advanced1D.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.advanced1D.setObjectName("advanced1D")
        self.horizontalLayout_5.addWidget(self.advanced1D)
        self.horizontalLayout_6.addLayout(self.horizontalLayout_5)
        self.frame1D_layout.addWidget(self.frame1D_buttons)
        self.horizontalLayout.addLayout(self.frame1D_layout)
        self.verticalLayout.addWidget(self.frame1D)
        self.frame2D = QtWidgets.QFrame(Form)
        self.frame2D.setMaximumSize(QtCore.QSize(16777215, 190))
        self.frame2D.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame2D.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame2D.setLineWidth(20)
        self.frame2D.setMidLineWidth(5)
        self.frame2D.setObjectName("frame2D")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.frame2D)
        self.horizontalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_12.setSpacing(0)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.frame2D_layout = QtWidgets.QVBoxLayout()
        self.frame2D_layout.setSpacing(0)
        self.frame2D_layout.setObjectName("frame2D_layout")
        self.frame2D_header = QtWidgets.QFrame(self.frame2D)
        self.frame2D_header.setMaximumSize(QtCore.QSize(16777215, 50))
        self.frame2D_header.setAutoFillBackground(True)
        self.frame2D_header.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame2D_header.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame2D_header.setObjectName("frame2D_header")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.frame2D_header)
        self.horizontalLayout_7.setContentsMargins(15, 3, 5, 3)
        self.horizontalLayout_7.setSpacing(0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label2D = QtWidgets.QLabel(self.frame2D_header)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label2D.sizePolicy().hasHeightForWidth())
        self.label2D.setSizePolicy(sizePolicy)
        self.label2D.setMinimumSize(QtCore.QSize(70, 0))
        self.label2D.setMaximumSize(QtCore.QSize(70, 16777215))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label2D.setFont(font)
        self.label2D.setAutoFillBackground(False)
        self.label2D.setStyleSheet("background-color: rgba(195, 195, 195, 200);")
        self.label2D.setAlignment(QtCore.Qt.AlignCenter)
        self.label2D.setObjectName("label2D")
        self.horizontalLayout_8.addWidget(self.label2D)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem2)
        self.axis2D = QtWidgets.QComboBox(self.frame2D_header)
        self.axis2D.setObjectName("axis2D")
        self.axis2D.addItem("")
        self.axis2D.addItem("")
        self.horizontalLayout_8.addWidget(self.axis2D)
        spacerItem3 = QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem3)
        self.label_npts_2D = QtWidgets.QLabel(self.frame2D_header)
        self.label_npts_2D.setMaximumSize(QtCore.QSize(40, 16777215))
        self.label_npts_2D.setIndent(-1)
        self.label_npts_2D.setObjectName("label_npts_2D")
        self.horizontalLayout_8.addWidget(self.label_npts_2D)
        self.npts_radial_2D = QtWidgets.QLineEdit(self.frame2D_header)
        self.npts_radial_2D.setMaximumSize(QtCore.QSize(55, 16777215))
        self.npts_radial_2D.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.npts_radial_2D.setObjectName("npts_radial_2D")
        self.horizontalLayout_8.addWidget(self.npts_radial_2D)
        self.npts_azim_2D = QtWidgets.QLineEdit(self.frame2D_header)
        self.npts_azim_2D.setMaximumSize(QtCore.QSize(55, 16777215))
        self.npts_azim_2D.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.npts_azim_2D.setObjectName("npts_azim_2D")
        self.horizontalLayout_8.addWidget(self.npts_azim_2D)
        self.horizontalLayout_7.addLayout(self.horizontalLayout_8)
        self.frame2D_layout.addWidget(self.frame2D_header)
        self.frame2D_range = QtWidgets.QFrame(self.frame2D)
        self.frame2D_range.setMaximumSize(QtCore.QSize(16777215, 71))
        self.frame2D_range.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame2D_range.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame2D_range.setObjectName("frame2D_range")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.frame2D_range)
        self.horizontalLayout_9.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.gridLayout_2D = QtWidgets.QGridLayout()
        self.gridLayout_2D.setHorizontalSpacing(6)
        self.gridLayout_2D.setObjectName("gridLayout_2D")
        self.radial_autoRange_2D = QtWidgets.QCheckBox(self.frame2D_range)
        self.radial_autoRange_2D.setChecked(True)
        self.radial_autoRange_2D.setObjectName("radial_autoRange_2D")
        self.gridLayout_2D.addWidget(self.radial_autoRange_2D, 0, 4, 1, 1)
        self.unit_2D = QtWidgets.QComboBox(self.frame2D_range)
        self.unit_2D.setMaximumSize(QtCore.QSize(90, 16777215))
        self.unit_2D.setObjectName("unit_2D")
        self.unit_2D.addItem("")
        self.unit_2D.addItem("")
        self.gridLayout_2D.addWidget(self.unit_2D, 0, 0, 1, 1)
        self.label_to1_2 = QtWidgets.QLabel(self.frame2D_range)
        self.label_to1_2.setMaximumSize(QtCore.QSize(100, 30))
        self.label_to1_2.setObjectName("label_to1_2")
        self.gridLayout_2D.addWidget(self.label_to1_2, 1, 2, 1, 1)
        self.azim_low_2D = QtWidgets.QLineEdit(self.frame2D_range)
        self.azim_low_2D.setObjectName("azim_low_2D")
        self.gridLayout_2D.addWidget(self.azim_low_2D, 1, 1, 1, 1)
        self.radial_low_2D = QtWidgets.QLineEdit(self.frame2D_range)
        self.radial_low_2D.setObjectName("radial_low_2D")
        self.gridLayout_2D.addWidget(self.radial_low_2D, 0, 1, 1, 1)
        self.azim_autoRange_2D = QtWidgets.QCheckBox(self.frame2D_range)
        self.azim_autoRange_2D.setChecked(True)
        self.azim_autoRange_2D.setObjectName("azim_autoRange_2D")
        self.gridLayout_2D.addWidget(self.azim_autoRange_2D, 1, 4, 1, 1)
        self.azim_high_2D = QtWidgets.QLineEdit(self.frame2D_range)
        self.azim_high_2D.setObjectName("azim_high_2D")
        self.gridLayout_2D.addWidget(self.azim_high_2D, 1, 3, 1, 1)
        self.label_to2_2 = QtWidgets.QLabel(self.frame2D_range)
        self.label_to2_2.setObjectName("label_to2_2")
        self.gridLayout_2D.addWidget(self.label_to2_2, 0, 2, 1, 1)
        self.label_azim_2D = QtWidgets.QLabel(self.frame2D_range)
        self.label_azim_2D.setMinimumSize(QtCore.QSize(55, 0))
        self.label_azim_2D.setMaximumSize(QtCore.QSize(100, 50))
        self.label_azim_2D.setAlignment(QtCore.Qt.AlignCenter)
        self.label_azim_2D.setObjectName("label_azim_2D")
        self.gridLayout_2D.addWidget(self.label_azim_2D, 1, 0, 1, 1)
        self.radial_high_2D = QtWidgets.QLineEdit(self.frame2D_range)
        self.radial_high_2D.setObjectName("radial_high_2D")
        self.gridLayout_2D.addWidget(self.radial_high_2D, 0, 3, 1, 1)
        self.horizontalLayout_9.addLayout(self.gridLayout_2D)
        self.frame2D_layout.addWidget(self.frame2D_range)
        self.frame2D_buttons = QtWidgets.QFrame(self.frame2D)
        self.frame2D_buttons.setMaximumSize(QtCore.QSize(16777215, 71))
        self.frame2D_buttons.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame2D_buttons.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame2D_buttons.setObjectName("frame2D_buttons")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.frame2D_buttons)
        self.horizontalLayout_10.setContentsMargins(5, 5, 5, 20)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setSpacing(6)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.integrate2D = QtWidgets.QPushButton(self.frame2D_buttons)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.integrate2D.sizePolicy().hasHeightForWidth())
        self.integrate2D.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(13)
        self.integrate2D.setFont(font)
        self.integrate2D.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.integrate2D.setObjectName("integrate2D")
        self.horizontalLayout_11.addWidget(self.integrate2D)
        self.advanced2D = QtWidgets.QPushButton(self.frame2D_buttons)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.advanced2D.sizePolicy().hasHeightForWidth())
        self.advanced2D.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(13)
        self.advanced2D.setFont(font)
        self.advanced2D.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.advanced2D.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.advanced2D.setObjectName("advanced2D")
        self.horizontalLayout_11.addWidget(self.advanced2D)
        self.horizontalLayout_10.addLayout(self.horizontalLayout_11)
        self.frame2D_layout.addWidget(self.frame2D_buttons)
        self.horizontalLayout_12.addLayout(self.frame2D_layout)
        self.verticalLayout.addWidget(self.frame2D)
        self.frame_3 = QtWidgets.QFrame(Form)
        self.frame_3.setMaximumSize(QtCore.QSize(16777215, 50))
        self.frame_3.setAutoFillBackground(True)
        self.frame_3.setStyleSheet("")
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_14.setContentsMargins(5, 0, 5, 0)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setSpacing(6)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.raw_to_tif = QtWidgets.QPushButton(self.frame_3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.raw_to_tif.setFont(font)
        self.raw_to_tif.setObjectName("raw_to_tif")
        self.horizontalLayout_13.addWidget(self.raw_to_tif)
        self.pyfai_calib = QtWidgets.QPushButton(self.frame_3)
        self.pyfai_calib.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.pyfai_calib.setFont(font)
        self.pyfai_calib.setObjectName("pyfai_calib")
        self.horizontalLayout_13.addWidget(self.pyfai_calib)
        self.get_mask = QtWidgets.QPushButton(self.frame_3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.get_mask.setFont(font)
        self.get_mask.setObjectName("get_mask")
        self.horizontalLayout_13.addWidget(self.get_mask)
        self.horizontalLayout_14.addLayout(self.horizontalLayout_13)
        self.verticalLayout.addWidget(self.frame_3)
        self.label_npts_1D.setBuddy(self.npts_1D)
        self.label_npts_2D.setBuddy(self.npts_radial_2D)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        Form.setTabOrder(self.axis1D, self.npts_1D)
        Form.setTabOrder(self.npts_1D, self.unit_1D)
        Form.setTabOrder(self.unit_1D, self.radial_low_1D)
        Form.setTabOrder(self.radial_low_1D, self.radial_high_1D)
        Form.setTabOrder(self.radial_high_1D, self.radial_autoRange_1D)
        Form.setTabOrder(self.radial_autoRange_1D, self.azim_low_1D)
        Form.setTabOrder(self.azim_low_1D, self.azim_high_1D)
        Form.setTabOrder(self.azim_high_1D, self.azim_autoRange_1D)
        Form.setTabOrder(self.azim_autoRange_1D, self.integrate1D)
        Form.setTabOrder(self.integrate1D, self.advanced1D)
        Form.setTabOrder(self.advanced1D, self.axis2D)
        Form.setTabOrder(self.axis2D, self.npts_radial_2D)
        Form.setTabOrder(self.npts_radial_2D, self.npts_azim_2D)
        Form.setTabOrder(self.npts_azim_2D, self.unit_2D)
        Form.setTabOrder(self.unit_2D, self.radial_low_2D)
        Form.setTabOrder(self.radial_low_2D, self.radial_high_2D)
        Form.setTabOrder(self.radial_high_2D, self.radial_autoRange_2D)
        Form.setTabOrder(self.radial_autoRange_2D, self.azim_low_2D)
        Form.setTabOrder(self.azim_low_2D, self.azim_high_2D)
        Form.setTabOrder(self.azim_high_2D, self.azim_autoRange_2D)
        Form.setTabOrder(self.azim_autoRange_2D, self.integrate2D)
        Form.setTabOrder(self.integrate2D, self.advanced2D)
        Form.setTabOrder(self.advanced2D, self.pyfai_calib)
        Form.setTabOrder(self.pyfai_calib, self.get_mask)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label1D.setText(_translate("Form", "1-D"))
        self.axis1D.setItemText(0, _translate("Form", "Radial"))
        self.label_npts_1D.setText(_translate("Form", "Points"))
        self.npts_1D.setText(_translate("Form", "3000"))
        self.npts_1D.setPlaceholderText(_translate("Form", "1000"))
        self.radial_autoRange_1D.setText(_translate("Form", "Auto"))
        self.unit_1D.setItemText(0, _translate("Form", "q (u\\u212Bu\\u207Bu\\u00B9)"))
        self.unit_1D.setItemText(1, _translate("Form", "2 u\\u03B8"))
        self.label_to1.setText(_translate("Form", "to"))
        self.azim_low_1D.setText(_translate("Form", "-180"))
        self.radial_low_1D.setText(_translate("Form", "0"))
        self.azim_autoRange_1D.setText(_translate("Form", "Auto"))
        self.azim_high_1D.setText(_translate("Form", "180"))
        self.label_to2.setText(_translate("Form", "to"))
        self.label_azim_1D.setText(_translate("Form", "Chi"))
        self.radial_high_1D.setText(_translate("Form", "5"))
        self.integrate1D.setText(_translate("Form", "Re-Integrate"))
        self.advanced1D.setText(_translate("Form", "Advanced..."))
        self.label2D.setText(_translate("Form", "2-D"))
        self.axis2D.setItemText(0, _translate("Form", "Q-Chi"))
        self.axis2D.setItemText(1, _translate("Form", "Qz-Qxy"))
        self.label_npts_2D.setText(_translate("Form", "Points"))
        self.npts_radial_2D.setText(_translate("Form", "500"))
        self.npts_radial_2D.setPlaceholderText(_translate("Form", "500"))
        self.npts_azim_2D.setText(_translate("Form", "500"))
        self.radial_autoRange_2D.setText(_translate("Form", "Auto"))
        self.unit_2D.setItemText(0, _translate("Form", "q (u\\u212Bu\\u207Bu\\u00B9)"))
        self.unit_2D.setItemText(1, _translate("Form", "2 u\\u03B8"))
        self.label_to1_2.setText(_translate("Form", "to"))
        self.azim_low_2D.setText(_translate("Form", "-180"))
        self.radial_low_2D.setText(_translate("Form", "0"))
        self.azim_autoRange_2D.setText(_translate("Form", "Auto"))
        self.azim_high_2D.setText(_translate("Form", "180"))
        self.label_to2_2.setText(_translate("Form", "to"))
        self.label_azim_2D.setText(_translate("Form", "Chi"))
        self.radial_high_2D.setText(_translate("Form", "5"))
        self.integrate2D.setText(_translate("Form", "Re-Integrate"))
        self.advanced2D.setText(_translate("Form", "Advanced..."))
        self.raw_to_tif.setText(_translate("Form", "Raw -> Tif"))
        self.pyfai_calib.setText(_translate("Form", "Calibrate"))
        self.get_mask.setText(_translate("Form", "Make Mask"))
