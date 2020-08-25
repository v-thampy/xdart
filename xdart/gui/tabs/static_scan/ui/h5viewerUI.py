# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'h5viewerUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1440, 828)
        self.layout = QtWidgets.QGridLayout(Form)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setVerticalSpacing(0)
        self.layout.setObjectName("layout")
        self.frame = QtWidgets.QFrame(Form)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.layout.addWidget(self.frame, 0, 0, 1, 1)
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.layout.addWidget(self.label, 1, 0, 1, 1)
        self.listScans = QtWidgets.QListWidget(Form)
        self.listScans.setAlternatingRowColors(True)
        self.listScans.setObjectName("listScans")
        self.layout.addWidget(self.listScans, 2, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.layout.addWidget(self.label_2, 1, 1, 1, 1)
        self.listData = QtWidgets.QListWidget(Form)
        self.listData.setEditTriggers(QtWidgets.QAbstractItemView.CurrentChanged|QtWidgets.QAbstractItemView.DoubleClicked|QtWidgets.QAbstractItemView.EditKeyPressed|QtWidgets.QAbstractItemView.SelectedClicked)
        self.listData.setTabKeyNavigation(True)
        self.listData.setAlternatingRowColors(False)
        self.listData.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listData.setSelectionRectVisible(True)
        self.listData.setObjectName("listData")
        self.layout.addWidget(self.listData, 2, 1, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Scans"))
        self.label_2.setText(_translate("Form", "Data"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
