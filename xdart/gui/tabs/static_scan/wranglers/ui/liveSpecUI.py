# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'liveSpecUI.ui'
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
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1440, 842)
        self.verticalLayout = QVBoxLayout(Form)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.paramFrame = QFrame(Form)
        self.paramFrame.setObjectName(u"paramFrame")
        self.paramFrame.setFrameShape(QFrame.StyledPanel)
        self.paramFrame.setFrameShadow(QFrame.Raised)

        self.verticalLayout.addWidget(self.paramFrame)

        self.specLabel = QLabel(Form)
        self.specLabel.setObjectName(u"specLabel")
        self.specLabel.setMaximumSize(QSize(16777215, 40))

        self.verticalLayout.addWidget(self.specLabel)

        self.commandFrame = QFrame(Form)
        self.commandFrame.setObjectName(u"commandFrame")
        self.commandFrame.setMaximumSize(QSize(16777215, 40))
        self.commandFrame.setFrameShape(QFrame.StyledPanel)
        self.commandFrame.setFrameShadow(QFrame.Raised)
        self.commandLayout = QHBoxLayout(self.commandFrame)
        self.commandLayout.setObjectName(u"commandLayout")

        self.verticalLayout.addWidget(self.commandFrame)

        self.frame = QFrame(Form)
        self.frame.setObjectName(u"frame")
        self.frame.setMaximumSize(QSize(16777215, 40))
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout = QHBoxLayout(self.frame)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.startButton = QPushButton(self.frame)
        self.startButton.setObjectName(u"startButton")

        self.horizontalLayout.addWidget(self.startButton)

        self.stopButton = QPushButton(self.frame)
        self.stopButton.setObjectName(u"stopButton")

        self.horizontalLayout.addWidget(self.stopButton)


        self.verticalLayout.addWidget(self.frame)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.specLabel.setText("")
        self.startButton.setText(QCoreApplication.translate("Form", u"Start", None))
        self.stopButton.setText(QCoreApplication.translate("Form", u"Stop", None))
    # retranslateUi

