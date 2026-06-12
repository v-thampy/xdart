# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'rangeWidgetUI.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFrame,
    QHBoxLayout, QLabel, QSizePolicy, QSpinBox,
    QVBoxLayout, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1440, 817)
        self.verticalLayout_2 = QVBoxLayout(Form)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.frame_3 = QFrame(Form)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.verticalLayout = QVBoxLayout(self.frame_3)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.titleLabel = QLabel(self.frame_3)
        self.titleLabel.setObjectName(u"titleLabel")
        font = QFont()
        font.setBold(True)
        self.titleLabel.setFont(font)
        self.titleLabel.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.titleLabel)

        self.frame = QFrame(self.frame_3)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.NoFrame)
        self.frame.setFrameShadow(QFrame.Raised)
        self.frame.setLineWidth(0)
        self.horizontalLayout = QHBoxLayout(self.frame)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.rangeLabel = QLabel(self.frame)
        self.rangeLabel.setObjectName(u"rangeLabel")
        self.rangeLabel.setFont(font)

        self.horizontalLayout.addWidget(self.rangeLabel)

        self.low = QDoubleSpinBox(self.frame)
        self.low.setObjectName(u"low")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.low.sizePolicy().hasHeightForWidth())
        self.low.setSizePolicy(sizePolicy)
        self.low.setMinimumSize(QSize(0, 0))
        self.low.setMaximumSize(QSize(68, 16777215))
        self.low.setMaximum(1000000.000000000000000)
        self.low.setSingleStep(0.100000000000000)
        self.low.setValue(0.000000000000000)

        self.horizontalLayout.addWidget(self.low)

        self.label_4 = QLabel(self.frame)
        self.label_4.setObjectName(u"label_4")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy1)
        self.label_4.setMaximumSize(QSize(20, 16777215))
        self.label_4.setAlignment(Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.label_4)

        self.high = QDoubleSpinBox(self.frame)
        self.high.setObjectName(u"high")
        sizePolicy.setHeightForWidth(self.high.sizePolicy().hasHeightForWidth())
        self.high.setSizePolicy(sizePolicy)
        self.high.setMaximumSize(QSize(68, 16777215))
        self.high.setSingleStep(0.100000000000000)

        self.horizontalLayout.addWidget(self.high)

        self.units = QComboBox(self.frame)
        self.units.setObjectName(u"units")
        self.units.setMaximumSize(QSize(68, 16777215))

        self.horizontalLayout.addWidget(self.units)


        self.verticalLayout.addWidget(self.frame)

        self.frame_2 = QFrame(self.frame_3)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.NoFrame)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.frame_2.setLineWidth(0)
        self.horizontalLayout_2 = QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.pointsLabel = QLabel(self.frame_2)
        self.pointsLabel.setObjectName(u"pointsLabel")
        self.pointsLabel.setFont(font)

        self.horizontalLayout_2.addWidget(self.pointsLabel)

        self.points = QSpinBox(self.frame_2)
        self.points.setObjectName(u"points")
        sizePolicy.setHeightForWidth(self.points.sizePolicy().hasHeightForWidth())
        self.points.setSizePolicy(sizePolicy)
        self.points.setMaximumSize(QSize(68, 16777215))
        self.points.setMinimum(1)
        self.points.setMaximum(10000000)

        self.horizontalLayout_2.addWidget(self.points)

        self.stepLabel = QLabel(self.frame_2)
        self.stepLabel.setObjectName(u"stepLabel")
        self.stepLabel.setMaximumSize(QSize(60, 16777215))
        self.stepLabel.setFont(font)

        self.horizontalLayout_2.addWidget(self.stepLabel)

        self.step = QDoubleSpinBox(self.frame_2)
        self.step.setObjectName(u"step")
        sizePolicy.setHeightForWidth(self.step.sizePolicy().hasHeightForWidth())
        self.step.setSizePolicy(sizePolicy)
        self.step.setMaximumSize(QSize(68, 16777215))
        self.step.setDecimals(3)
        self.step.setMinimum(0.001000000000000)
        self.step.setMaximum(1000000.000000000000000)
        self.step.setSingleStep(0.100000000000000)

        self.horizontalLayout_2.addWidget(self.step)


        self.verticalLayout.addWidget(self.frame_2)


        self.verticalLayout_2.addWidget(self.frame_3)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.titleLabel.setText(QCoreApplication.translate("Form", u"Title", None))
        self.rangeLabel.setText(QCoreApplication.translate("Form", u"Range", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"to", None))
        self.pointsLabel.setText(QCoreApplication.translate("Form", u"Points", None))
        self.stepLabel.setText(QCoreApplication.translate("Form", u"Step Size", None))
    # retranslateUi

