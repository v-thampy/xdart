# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'imageWidgetUI.ui'
##
## Created by: Qt User Interface Compiler version 6.10.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGridLayout,
    QHBoxLayout, QSizePolicy, QToolButton, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(688, 480)
        self.horizontalLayout = QHBoxLayout(Form)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.imageFrame = QFrame(Form)
        self.imageFrame.setObjectName(u"imageFrame")
        self.imageFrame.setFrameShape(QFrame.NoFrame)
        self.imageFrame.setFrameShadow(QFrame.Plain)
        self.imageFrame.setLineWidth(0)

        self.horizontalLayout.addWidget(self.imageFrame)

        self.toolFrame = QFrame(Form)
        self.toolFrame.setObjectName(u"toolFrame")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolFrame.sizePolicy().hasHeightForWidth())
        self.toolFrame.setSizePolicy(sizePolicy)
        self.toolFrame.setMinimumSize(QSize(120, 0))
        self.toolFrame.setFrameShape(QFrame.NoFrame)
        self.toolFrame.setFrameShadow(QFrame.Plain)
        self.toolFrame.setLineWidth(0)
        self.toolLayout = QGridLayout(self.toolFrame)
        self.toolLayout.setObjectName(u"toolLayout")
        self.toolLayout.setContentsMargins(1, 1, 1, 1)
        self.cmapBox = QComboBox(self.toolFrame)
        self.cmapBox.addItem("")
        self.cmapBox.addItem("")
        self.cmapBox.addItem("")
        self.cmapBox.addItem("")
        self.cmapBox.addItem("")
        self.cmapBox.addItem("")
        self.cmapBox.addItem("")
        self.cmapBox.addItem("")
        self.cmapBox.addItem("")
        self.cmapBox.addItem("")
        self.cmapBox.addItem("")
        self.cmapBox.addItem("")
        self.cmapBox.setObjectName(u"cmapBox")

        self.toolLayout.addWidget(self.cmapBox, 0, 1, 1, 1)

        self.logButton = QToolButton(self.toolFrame)
        self.logButton.setObjectName(u"logButton")
        self.logButton.setMinimumSize(QSize(30, 0))
        self.logButton.setMaximumSize(QSize(30, 16777215))
        self.logButton.setCheckable(True)

        self.toolLayout.addWidget(self.logButton, 0, 0, 1, 1)


        self.horizontalLayout.addWidget(self.toolFrame)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.cmapBox.setItemText(0, QCoreApplication.translate("Form", u"grey", None))
        self.cmapBox.setItemText(1, QCoreApplication.translate("Form", u"viridis", None))
        self.cmapBox.setItemText(2, QCoreApplication.translate("Form", u"plasma", None))
        self.cmapBox.setItemText(3, QCoreApplication.translate("Form", u"inferno", None))
        self.cmapBox.setItemText(4, QCoreApplication.translate("Form", u"magma", None))
        self.cmapBox.setItemText(5, QCoreApplication.translate("Form", u"spectrum", None))
        self.cmapBox.setItemText(6, QCoreApplication.translate("Form", u"thermal", None))
        self.cmapBox.setItemText(7, QCoreApplication.translate("Form", u"flame", None))
        self.cmapBox.setItemText(8, QCoreApplication.translate("Form", u"yellowy", None))
        self.cmapBox.setItemText(9, QCoreApplication.translate("Form", u"bipolar", None))
        self.cmapBox.setItemText(10, QCoreApplication.translate("Form", u"cyclic", None))
        self.cmapBox.setItemText(11, QCoreApplication.translate("Form", u"greyclip", None))

        self.logButton.setText(QCoreApplication.translate("Form", u"Log", None))
    # retranslateUi

