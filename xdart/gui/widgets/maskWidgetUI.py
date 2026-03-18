# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'maskWidgetUI.ui'
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
    QHBoxLayout, QLabel, QPushButton, QRadioButton,
    QSizePolicy, QSlider, QSpacerItem, QSpinBox,
    QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(363, 366)
        self.horizontalLayout = QHBoxLayout(Form)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.toolsFrame = QFrame(Form)
        self.toolsFrame.setObjectName(u"toolsFrame")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolsFrame.sizePolicy().hasHeightForWidth())
        self.toolsFrame.setSizePolicy(sizePolicy)
        self.toolsFrame.setFrameShape(QFrame.StyledPanel)
        self.toolsFrame.setFrameShadow(QFrame.Raised)
        self.gridLayout = QGridLayout(self.toolsFrame)
        self.gridLayout.setObjectName(u"gridLayout")
        self.lineMaskButton = QPushButton(self.toolsFrame)
        self.lineMaskButton.setObjectName(u"lineMaskButton")

        self.gridLayout.addWidget(self.lineMaskButton, 3, 0, 1, 1)

        self.frame = QFrame(self.toolsFrame)
        self.frame.setObjectName(u"frame")
        self.frame.setMinimumSize(QSize(150, 150))
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.gridLayout_2 = QGridLayout(self.frame)
        self.gridLayout_2.setSpacing(1)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(1, 1, 1, 1)
        self.horizontalSpacer_3 = QSpacerItem(88, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer_3, 2, 2, 1, 1)

        self.xMinBox = QSpinBox(self.frame)
        self.xMinBox.setObjectName(u"xMinBox")

        self.gridLayout_2.addWidget(self.xMinBox, 3, 2, 1, 1)

        self.horizontalSpacer_4 = QSpacerItem(88, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer_4, 3, 1, 1, 1)

        self.yMinBox = QSpinBox(self.frame)
        self.yMinBox.setObjectName(u"yMinBox")

        self.gridLayout_2.addWidget(self.yMinBox, 2, 1, 1, 1)

        self.xMaxBox = QSpinBox(self.frame)
        self.xMaxBox.setObjectName(u"xMaxBox")

        self.gridLayout_2.addWidget(self.xMaxBox, 1, 2, 1, 1)

        self.horizontalSpacer_5 = QSpacerItem(88, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer_5, 3, 3, 1, 1)

        self.horizontalSpacer = QSpacerItem(88, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer, 1, 1, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(88, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer_2, 1, 3, 1, 1)

        self.yMaxBox = QSpinBox(self.frame)
        self.yMaxBox.setObjectName(u"yMaxBox")

        self.gridLayout_2.addWidget(self.yMaxBox, 2, 3, 1, 1)

        self.label = QLabel(self.frame)
        self.label.setObjectName(u"label")

        self.gridLayout_2.addWidget(self.label, 4, 0, 1, 1)

        self.label_2 = QLabel(self.frame)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout_2.addWidget(self.label_2, 0, 4, 1, 1)

        self.yMaxSlider = QSlider(self.frame)
        self.yMaxSlider.setObjectName(u"yMaxSlider")
        self.yMaxSlider.setOrientation(Qt.Vertical)

        self.gridLayout_2.addWidget(self.yMaxSlider, 1, 4, 3, 1)

        self.yMinSlider = QSlider(self.frame)
        self.yMinSlider.setObjectName(u"yMinSlider")
        self.yMinSlider.setOrientation(Qt.Vertical)

        self.gridLayout_2.addWidget(self.yMinSlider, 1, 0, 3, 1)

        self.xMaxSlider = QSlider(self.frame)
        self.xMaxSlider.setObjectName(u"xMaxSlider")
        self.xMaxSlider.setOrientation(Qt.Horizontal)

        self.gridLayout_2.addWidget(self.xMaxSlider, 0, 1, 1, 3)

        self.xMinSlider = QSlider(self.frame)
        self.xMinSlider.setObjectName(u"xMinSlider")
        self.xMinSlider.setOrientation(Qt.Horizontal)

        self.gridLayout_2.addWidget(self.xMinSlider, 4, 1, 1, 3)


        self.gridLayout.addWidget(self.frame, 4, 0, 1, 2)

        self.ellipseMaskButton = QPushButton(self.toolsFrame)
        self.ellipseMaskButton.setObjectName(u"ellipseMaskButton")

        self.gridLayout.addWidget(self.ellipseMaskButton, 2, 1, 1, 1)

        self.addROI = QRadioButton(self.toolsFrame)
        self.addROI.setObjectName(u"addROI")
        self.addROI.setChecked(True)

        self.gridLayout.addWidget(self.addROI, 1, 0, 1, 1)

        self.rectMaskButton = QPushButton(self.toolsFrame)
        self.rectMaskButton.setObjectName(u"rectMaskButton")

        self.gridLayout.addWidget(self.rectMaskButton, 2, 0, 1, 1)

        self.subtractROI = QRadioButton(self.toolsFrame)
        self.subtractROI.setObjectName(u"subtractROI")

        self.gridLayout.addWidget(self.subtractROI, 1, 1, 1, 1)

        self.polyMaskButton = QPushButton(self.toolsFrame)
        self.polyMaskButton.setObjectName(u"polyMaskButton")

        self.gridLayout.addWidget(self.polyMaskButton, 3, 1, 1, 1)

        self.archList = QComboBox(self.toolsFrame)
        self.archList.setObjectName(u"archList")

        self.gridLayout.addWidget(self.archList, 0, 0, 1, 1)

        self.clearButton = QPushButton(self.toolsFrame)
        self.clearButton.setObjectName(u"clearButton")

        self.gridLayout.addWidget(self.clearButton, 0, 1, 1, 1)

        self.setCurrent = QPushButton(self.toolsFrame)
        self.setCurrent.setObjectName(u"setCurrent")

        self.gridLayout.addWidget(self.setCurrent, 5, 0, 1, 1)

        self.setGlobal = QPushButton(self.toolsFrame)
        self.setGlobal.setObjectName(u"setGlobal")

        self.gridLayout.addWidget(self.setGlobal, 5, 1, 1, 1)


        self.horizontalLayout.addWidget(self.toolsFrame)


        self.retranslateUi(Form)
        self.xMaxBox.valueChanged.connect(self.xMaxSlider.setValue)
        self.xMaxSlider.sliderMoved.connect(self.xMaxBox.setValue)
        self.yMaxBox.valueChanged.connect(self.yMaxSlider.setValue)
        self.yMaxSlider.sliderMoved.connect(self.yMaxBox.setValue)
        self.xMinBox.valueChanged.connect(self.xMinSlider.setValue)
        self.xMinSlider.sliderMoved.connect(self.xMinBox.setValue)
        self.yMinSlider.sliderMoved.connect(self.yMinBox.setValue)
        self.yMinBox.valueChanged.connect(self.yMinSlider.setValue)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.lineMaskButton.setText(QCoreApplication.translate("Form", u"Line", None))
        self.label.setText(QCoreApplication.translate("Form", u"Min", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"Max", None))
        self.ellipseMaskButton.setText(QCoreApplication.translate("Form", u"Ellipse", None))
        self.addROI.setText(QCoreApplication.translate("Form", u"Add", None))
        self.rectMaskButton.setText(QCoreApplication.translate("Form", u"Rect", None))
        self.subtractROI.setText(QCoreApplication.translate("Form", u"Subtract", None))
        self.polyMaskButton.setText(QCoreApplication.translate("Form", u"Poly", None))
        self.clearButton.setText(QCoreApplication.translate("Form", u"Clear", None))
        self.setCurrent.setText(QCoreApplication.translate("Form", u"Set Current", None))
        self.setGlobal.setText(QCoreApplication.translate("Form", u"Set Global", None))
    # retranslateUi

