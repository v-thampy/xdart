# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'imageWidgetUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1440, 842)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.imageFrame = QtWidgets.QFrame(Form)
        self.imageFrame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.imageFrame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.imageFrame.setLineWidth(0)
        self.imageFrame.setObjectName("imageFrame")
        self.horizontalLayout.addWidget(self.imageFrame)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
