# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'staticUI.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QHBoxLayout,
    QSizePolicy, QSplitter, QStackedWidget, QVBoxLayout,
    QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1440, 842)
        self.horizontalLayout = QHBoxLayout(Form)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.mainSplitter = QSplitter(Form)
        self.mainSplitter.setObjectName(u"mainSplitter")
        self.mainSplitter.setOrientation(Qt.Horizontal)
        self.leftFrame = QFrame(self.mainSplitter)
        self.leftFrame.setObjectName(u"leftFrame")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.leftFrame.sizePolicy().hasHeightForWidth())
        self.leftFrame.setSizePolicy(sizePolicy)
        self.leftFrame.setMinimumSize(QSize(200, 0))
        self.leftFrame.setFrameShape(QFrame.StyledPanel)
        self.leftFrame.setFrameShadow(QFrame.Raised)
        self.leftFrame.setLineWidth(5)
        self.horizontalLayout_4 = QHBoxLayout(self.leftFrame)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.leftSplitter = QSplitter(self.leftFrame)
        self.leftSplitter.setObjectName(u"leftSplitter")
        self.leftSplitter.setOrientation(Qt.Vertical)
        self.hdf5Frame = QFrame(self.leftSplitter)
        self.hdf5Frame.setObjectName(u"hdf5Frame")
        self.hdf5Frame.setMinimumSize(QSize(0, 0))
        self.hdf5Frame.setFrameShape(QFrame.StyledPanel)
        self.hdf5Frame.setFrameShadow(QFrame.Raised)
        self.hdf5Frame.setLineWidth(3)
        self.leftSplitter.addWidget(self.hdf5Frame)
        self.metaFrame = QFrame(self.leftSplitter)
        self.metaFrame.setObjectName(u"metaFrame")
        self.metaFrame.setFrameShape(QFrame.StyledPanel)
        self.metaFrame.setFrameShadow(QFrame.Raised)
        self.metaFrame.setLineWidth(3)
        self.leftSplitter.addWidget(self.metaFrame)

        self.horizontalLayout_4.addWidget(self.leftSplitter)

        self.mainSplitter.addWidget(self.leftFrame)
        self.middleFrame = QFrame(self.mainSplitter)
        self.middleFrame.setObjectName(u"middleFrame")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(2)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.middleFrame.sizePolicy().hasHeightForWidth())
        self.middleFrame.setSizePolicy(sizePolicy1)
        self.middleFrame.setFrameShape(QFrame.StyledPanel)
        self.middleFrame.setFrameShadow(QFrame.Raised)
        self.middleFrame.setLineWidth(5)
        self.mainSplitter.addWidget(self.middleFrame)
        self.rightFrame = QFrame(self.mainSplitter)
        self.rightFrame.setObjectName(u"rightFrame")
        sizePolicy.setHeightForWidth(self.rightFrame.sizePolicy().hasHeightForWidth())
        self.rightFrame.setSizePolicy(sizePolicy)
        self.rightFrame.setFrameShape(QFrame.StyledPanel)
        self.rightFrame.setFrameShadow(QFrame.Raised)
        self.rightFrame.setLineWidth(5)
        self.horizontalLayout_3 = QHBoxLayout(self.rightFrame)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.rightSplitter = QSplitter(self.rightFrame)
        self.rightSplitter.setObjectName(u"rightSplitter")
        self.rightSplitter.setOrientation(Qt.Vertical)
        self.integratorFrame = QFrame(self.rightSplitter)
        self.integratorFrame.setObjectName(u"integratorFrame")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.integratorFrame.sizePolicy().hasHeightForWidth())
        self.integratorFrame.setSizePolicy(sizePolicy2)
        self.integratorFrame.setMaximumSize(QSize(16777215, 400))
        self.integratorFrame.setFrameShape(QFrame.StyledPanel)
        self.integratorFrame.setFrameShadow(QFrame.Raised)
        self.integratorFrame.setLineWidth(3)
        self.rightSplitter.addWidget(self.integratorFrame)
        self.wranglerFrame = QFrame(self.rightSplitter)
        self.wranglerFrame.setObjectName(u"wranglerFrame")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.wranglerFrame.sizePolicy().hasHeightForWidth())
        self.wranglerFrame.setSizePolicy(sizePolicy3)
        self.wranglerFrame.setFrameShape(QFrame.StyledPanel)
        self.wranglerFrame.setFrameShadow(QFrame.Raised)
        self.wranglerFrame.setLineWidth(3)
        self.verticalLayout = QVBoxLayout(self.wranglerFrame)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.wranglerBox = QComboBox(self.wranglerFrame)
        self.wranglerBox.setObjectName(u"wranglerBox")
        self.wranglerBox.setFocusPolicy(Qt.ClickFocus)

        self.verticalLayout.addWidget(self.wranglerBox)

        self.wranglerStack = QStackedWidget(self.wranglerFrame)
        self.wranglerStack.setObjectName(u"wranglerStack")
        sizePolicy3.setHeightForWidth(self.wranglerStack.sizePolicy().hasHeightForWidth())
        self.wranglerStack.setSizePolicy(sizePolicy3)

        self.verticalLayout.addWidget(self.wranglerStack)

        self.rightSplitter.addWidget(self.wranglerFrame)

        self.horizontalLayout_3.addWidget(self.rightSplitter)

        self.mainSplitter.addWidget(self.rightFrame)

        self.horizontalLayout.addWidget(self.mainSplitter)


        self.retranslateUi(Form)
        self.wranglerBox.currentIndexChanged.connect(self.wranglerStack.setCurrentIndex)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
    # retranslateUi

