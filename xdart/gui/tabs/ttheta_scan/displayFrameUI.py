# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'displayFrameUI.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(954, 442)
        self.layout = QtWidgets.QHBoxLayout(Form)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.setObjectName("layout")
        self.frame = QtWidgets.QFrame(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setMinimumSize(QtCore.QSize(40, 0))
        self.frame.setMaximumSize(QtCore.QSize(40, 16777215))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_2.setContentsMargins(4, 0, 4, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushLeft = QtWidgets.QPushButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushLeft.sizePolicy().hasHeightForWidth())
        self.pushLeft.setSizePolicy(sizePolicy)
        self.pushLeft.setMinimumSize(QtCore.QSize(30, 0))
        self.pushLeft.setMaximumSize(QtCore.QSize(30, 16777215))
        self.pushLeft.setObjectName("pushLeft")
        self.verticalLayout_2.addWidget(self.pushLeft)
        self.pushLeftLast = QtWidgets.QPushButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushLeftLast.sizePolicy().hasHeightForWidth())
        self.pushLeftLast.setSizePolicy(sizePolicy)
        self.pushLeftLast.setMinimumSize(QtCore.QSize(30, 0))
        self.pushLeftLast.setMaximumSize(QtCore.QSize(30, 16777215))
        self.pushLeftLast.setObjectName("pushLeftLast")
        self.verticalLayout_2.addWidget(self.pushLeftLast)
        self.layout.addWidget(self.frame)
        self.splitter = QtWidgets.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.imageWindow = QtWidgets.QFrame(self.splitter)
        self.imageWindow.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.imageWindow.setFrameShadow(QtWidgets.QFrame.Raised)
        self.imageWindow.setLineWidth(3)
        self.imageWindow.setObjectName("imageWindow")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.imageWindow)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.labelCurrent = QtWidgets.QLabel(self.imageWindow)
        self.labelCurrent.setMinimumSize(QtCore.QSize(0, 20))
        self.labelCurrent.setMaximumSize(QtCore.QSize(16777215, 20))
        self.labelCurrent.setAlignment(QtCore.Qt.AlignCenter)
        self.labelCurrent.setObjectName("labelCurrent")
        self.verticalLayout_3.addWidget(self.labelCurrent)
        self.imageFrame = QtWidgets.QFrame(self.imageWindow)
        self.imageFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.imageFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.imageFrame.setObjectName("imageFrame")
        self.verticalLayout_3.addWidget(self.imageFrame)
        self.imageToolbar = QtWidgets.QFrame(self.imageWindow)
        self.imageToolbar.setMaximumSize(QtCore.QSize(16777215, 40))
        self.imageToolbar.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.imageToolbar.setFrameShadow(QtWidgets.QFrame.Raised)
        self.imageToolbar.setObjectName("imageToolbar")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.imageToolbar)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.imageIntRaw = QtWidgets.QComboBox(self.imageToolbar)
        self.imageIntRaw.setObjectName("imageIntRaw")
        self.imageIntRaw.addItem("")
        self.imageIntRaw.addItem("")
        self.horizontalLayout_2.addWidget(self.imageIntRaw)
        self.imageMethod = QtWidgets.QComboBox(self.imageToolbar)
        self.imageMethod.setObjectName("imageMethod")
        self.imageMethod.addItem("")
        self.imageMethod.addItem("")
        self.horizontalLayout_2.addWidget(self.imageMethod)
        self.imageUnit = QtWidgets.QComboBox(self.imageToolbar)
        self.imageUnit.setObjectName("imageUnit")
        self.imageUnit.addItem("")
        self.imageUnit.addItem("")
        self.horizontalLayout_2.addWidget(self.imageUnit)
        self.imageNRP = QtWidgets.QComboBox(self.imageToolbar)
        self.imageNRP.setObjectName("imageNRP")
        self.imageNRP.addItem("")
        self.imageNRP.addItem("")
        self.imageNRP.addItem("")
        self.horizontalLayout_2.addWidget(self.imageNRP)
        self.imageMask = QtWidgets.QCheckBox(self.imageToolbar)
        self.imageMask.setObjectName("imageMask")
        self.horizontalLayout_2.addWidget(self.imageMask)
        self.setMaskButton = QtWidgets.QPushButton(self.imageToolbar)
        self.setMaskButton.setObjectName("setMaskButton")
        self.horizontalLayout_2.addWidget(self.setMaskButton)
        self.shareAxis = QtWidgets.QCheckBox(self.imageToolbar)
        self.shareAxis.setObjectName("shareAxis")
        self.horizontalLayout_2.addWidget(self.shareAxis)
        self.verticalLayout_3.addWidget(self.imageToolbar)
        self.plotWindow = QtWidgets.QFrame(self.splitter)
        self.plotWindow.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.plotWindow.setFrameShadow(QtWidgets.QFrame.Raised)
        self.plotWindow.setLineWidth(3)
        self.plotWindow.setObjectName("plotWindow")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.plotWindow)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.plotFrame = QtWidgets.QFrame(self.plotWindow)
        self.plotFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.plotFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.plotFrame.setObjectName("plotFrame")
        self.verticalLayout_4.addWidget(self.plotFrame)
        self.plotToolBar = QtWidgets.QFrame(self.plotWindow)
        self.plotToolBar.setMaximumSize(QtCore.QSize(16777215, 40))
        self.plotToolBar.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.plotToolBar.setFrameShadow(QtWidgets.QFrame.Raised)
        self.plotToolBar.setObjectName("plotToolBar")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.plotToolBar)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.plotMethod = QtWidgets.QComboBox(self.plotToolBar)
        self.plotMethod.setObjectName("plotMethod")
        self.plotMethod.addItem("")
        self.plotMethod.addItem("")
        self.horizontalLayout.addWidget(self.plotMethod)
        self.plotUnit = QtWidgets.QComboBox(self.plotToolBar)
        self.plotUnit.setObjectName("plotUnit")
        self.plotUnit.addItem("")
        self.plotUnit.addItem("")
        self.horizontalLayout.addWidget(self.plotUnit)
        self.plotNRP = QtWidgets.QComboBox(self.plotToolBar)
        self.plotNRP.setObjectName("plotNRP")
        self.plotNRP.addItem("")
        self.plotNRP.addItem("")
        self.plotNRP.addItem("")
        self.horizontalLayout.addWidget(self.plotNRP)
        self.plotOverlay = QtWidgets.QCheckBox(self.plotToolBar)
        self.plotOverlay.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.plotOverlay.setObjectName("plotOverlay")
        self.horizontalLayout.addWidget(self.plotOverlay)
        self.verticalLayout_4.addWidget(self.plotToolBar)
        self.layout.addWidget(self.splitter)
        self.frame_2 = QtWidgets.QFrame(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setMinimumSize(QtCore.QSize(40, 0))
        self.frame_2.setMaximumSize(QtCore.QSize(40, 16777215))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout.setContentsMargins(4, 0, 4, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushRight = QtWidgets.QPushButton(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushRight.sizePolicy().hasHeightForWidth())
        self.pushRight.setSizePolicy(sizePolicy)
        self.pushRight.setMinimumSize(QtCore.QSize(30, 0))
        self.pushRight.setMaximumSize(QtCore.QSize(30, 16777215))
        self.pushRight.setObjectName("pushRight")
        self.verticalLayout.addWidget(self.pushRight)
        self.pushRightLast = QtWidgets.QPushButton(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushRightLast.sizePolicy().hasHeightForWidth())
        self.pushRightLast.setSizePolicy(sizePolicy)
        self.pushRightLast.setMinimumSize(QtCore.QSize(30, 0))
        self.pushRightLast.setMaximumSize(QtCore.QSize(30, 16777215))
        self.pushRightLast.setObjectName("pushRightLast")
        self.verticalLayout.addWidget(self.pushRightLast)
        self.layout.addWidget(self.frame_2)

        self.retranslateUi(Form)
        self.imageMethod.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushLeft.setText(_translate("Form", "<"))
        self.pushLeftLast.setText(_translate("Form", "<<"))
        self.labelCurrent.setText(_translate("Form", "Current"))
        self.imageIntRaw.setItemText(0, _translate("Form", "Integrated"))
        self.imageIntRaw.setItemText(1, _translate("Form", "Raw"))
        self.imageMethod.setItemText(0, _translate("Form", "Multi. Geo."))
        self.imageMethod.setItemText(1, _translate("Form", "By image"))
        self.imageUnit.setItemText(0, _translate("Form", "2 u038"))
        self.imageUnit.setItemText(1, _translate("Form", "q (A-1)"))
        self.imageNRP.setItemText(0, _translate("Form", "Normalized"))
        self.imageNRP.setItemText(1, _translate("Form", "Raw"))
        self.imageNRP.setItemText(2, _translate("Form", "Pixel Count"))
        self.imageMask.setText(_translate("Form", "Mask"))
        self.setMaskButton.setText(_translate("Form", "Set mask..."))
        self.shareAxis.setText(_translate("Form", "Share Axis"))
        self.plotMethod.setItemText(0, _translate("Form", "Multi. Geo."))
        self.plotMethod.setItemText(1, _translate("Form", "By image"))
        self.plotUnit.setItemText(0, _translate("Form", "2 u\\u03B8"))
        self.plotUnit.setItemText(1, _translate("Form", "q (A-1)"))
        self.plotNRP.setItemText(0, _translate("Form", "Normalized"))
        self.plotNRP.setItemText(1, _translate("Form", "Raw"))
        self.plotNRP.setItemText(2, _translate("Form", "Pixel Count"))
        self.plotOverlay.setText(_translate("Form", "Overlay"))
        self.pushRight.setText(_translate("Form", ">"))
        self.pushRightLast.setText(_translate("Form", ">>"))
