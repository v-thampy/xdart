# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'specUI.ui'
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
import os as _os
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1440, 369)
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
        self.specLabel.setMinimumSize(QSize(0, 30))
        self.specLabel.setMaximumSize(QSize(16777215, 40))

        self.verticalLayout.addWidget(self.specLabel)

        self.commandFrame = QFrame(Form)
        self.commandFrame.setObjectName(u"commandFrame")
        self.commandFrame.setMinimumSize(QSize(0, 40))
        self.commandFrame.setMaximumSize(QSize(16777215, 40))
        self.commandFrame.setFrameShape(QFrame.StyledPanel)
        self.commandFrame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.commandFrame)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(-1, 2, -1, 2)
        self.commandLayout = QHBoxLayout()
        self.commandLayout.setObjectName(u"commandLayout")
        self.advancedButton = QPushButton(self.commandFrame)
        self.advancedButton.setObjectName(u"advancedButton")
        self.advancedButton.setMaximumSize(QSize(80, 16777215))
        self.commandLayout.addWidget(self.advancedButton)
        self.modeLabel = QLabel(self.commandFrame)
        self.modeLabel.setObjectName(u"modeLabel")
        self.modeLabel.setVisible(False)  # hidden — no longer needed
        self.processingModeCombo = QComboBox(self.commandFrame)
        self.processingModeCombo.setObjectName(u"processingModeCombo")
        self.processingModeCombo.addItems([
            "Int 1D",
            "Int 2D",
            "Int 1D (XYE)",
            "Image Viewer",
            "XYE Viewer",
        ])
        self.commandLayout.addWidget(self.processingModeCombo)

        self.liveCheckBox = QCheckBox(self.commandFrame)
        self.liveCheckBox.setObjectName(u"liveCheckBox")
        self.liveCheckBox.setText(u"Live")
        self.commandLayout.addWidget(self.liveCheckBox)

        self.batchCheckBox = QCheckBox(self.commandFrame)
        self.batchCheckBox.setObjectName(u"batchCheckBox")
        self.batchCheckBox.setText(u"Batch")
        self.batchCheckBox.setChecked(True)
        self.commandLayout.addWidget(self.batchCheckBox)

        self.coresLabel = QLabel(self.commandFrame)
        self.coresLabel.setObjectName(u"coresLabel")
        self.commandLayout.addWidget(self.coresLabel)
        self.maxCoresSpinBox = QSpinBox(self.commandFrame)
        self.maxCoresSpinBox.setObjectName(u"maxCoresSpinBox")
        _cpu = _os.cpu_count() or 4
        self.maxCoresSpinBox.setMinimum(1)
        self.maxCoresSpinBox.setMaximum(_cpu)
        self.maxCoresSpinBox.setValue(min(_cpu - 1, 8) or 1)
        self.commandLayout.addWidget(self.maxCoresSpinBox)

        self.horizontalLayout_2.addLayout(self.commandLayout)


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
        self.startButton.setFocusPolicy(Qt.ClickFocus)

        self.horizontalLayout.addWidget(self.startButton)

        self.pauseButton = QPushButton(self.frame)
        self.pauseButton.setObjectName(u"pauseButton")
        self.pauseButton.setFocusPolicy(Qt.ClickFocus)

        self.horizontalLayout.addWidget(self.pauseButton)

        self.continueButton = QPushButton(self.frame)
        self.continueButton.setObjectName(u"continueButton")
        self.continueButton.setFocusPolicy(Qt.ClickFocus)

        self.horizontalLayout.addWidget(self.continueButton)

        self.stopButton = QPushButton(self.frame)
        self.stopButton.setObjectName(u"stopButton")
        self.stopButton.setFocusPolicy(Qt.ClickFocus)

        self.horizontalLayout.addWidget(self.stopButton)


        self.verticalLayout.addWidget(self.frame)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.specLabel.setText("")
        self.advancedButton.setText(QCoreApplication.translate("Form", u"Advanced...", None))
        self.modeLabel.setText(QCoreApplication.translate("Form", u"Mode:", None))
        self.coresLabel.setText(QCoreApplication.translate("Form", u"Cores:", None))
        self.startButton.setText(QCoreApplication.translate("Form", u"Start", None))
        self.pauseButton.setText(QCoreApplication.translate("Form", u"Pause", None))
        self.continueButton.setText(QCoreApplication.translate("Form", u"Continue", None))
        self.stopButton.setText(QCoreApplication.translate("Form", u"Stop", None))
    # retranslateUi

