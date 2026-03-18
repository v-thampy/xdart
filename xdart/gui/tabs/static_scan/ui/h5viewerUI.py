# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'h5viewerUI.ui'
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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QFrame, QGridLayout,
    QLabel, QListView, QListWidget, QListWidgetItem,
    QPushButton, QSizePolicy, QSplitter, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1440, 842)
        Form.setMinimumSize(QSize(0, 0))
        self.gridLayout = QGridLayout(Form)
#ifndef Q_OS_MAC
        self.gridLayout.setSpacing(-1)
#endif
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(8, 8, 8, 12)
        self.label_3 = QLabel(Form)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMaximumSize(QSize(16777215, 20))

        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)

        self.label_4 = QLabel(Form)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMaximumSize(QSize(70, 16777215))
        self.label_4.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_4, 1, 1, 1, 1)

        self.show_all = QPushButton(Form)
        self.show_all.setObjectName(u"show_all")
        self.show_all.setMaximumSize(QSize(16777215, 25))

        self.gridLayout.addWidget(self.show_all, 3, 0, 1, 1)

        self.auto_last = QPushButton(Form)
        self.auto_last.setObjectName(u"auto_last")
        self.auto_last.setMaximumSize(QSize(16777215, 25))

        self.gridLayout.addWidget(self.auto_last, 3, 1, 1, 1)

        self.splitter = QSplitter(Form)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setMinimumSize(QSize(0, 0))
        self.splitter.setOrientation(Qt.Horizontal)
        self.listScans = QListWidget(self.splitter)
        self.listScans.setObjectName(u"listScans")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.listScans.sizePolicy().hasHeightForWidth())
        self.listScans.setSizePolicy(sizePolicy)
        self.listScans.setMinimumSize(QSize(30, 0))
        self.listScans.setResizeMode(QListView.Adjust)
        self.splitter.addWidget(self.listScans)
        self.listData = QListWidget(self.splitter)
        self.listData.setObjectName(u"listData")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.listData.sizePolicy().hasHeightForWidth())
        self.listData.setSizePolicy(sizePolicy1)
        self.listData.setMinimumSize(QSize(45, 0))
        self.listData.setMaximumSize(QSize(60, 16777215))
        self.listData.setFocusPolicy(Qt.StrongFocus)
        self.listData.setTabKeyNavigation(False)
        self.listData.setAlternatingRowColors(True)
        self.listData.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.splitter.addWidget(self.listData)

        self.gridLayout.addWidget(self.splitter, 2, 0, 1, 2)

        self.frame = QFrame(Form)
        self.frame.setObjectName(u"frame")
        self.frame.setMaximumSize(QSize(16777215, 20))
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)

        self.gridLayout.addWidget(self.frame, 0, 0, 1, 2)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"  Scans", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"Data", None))
        self.show_all.setText(QCoreApplication.translate("Form", u"Show All", None))
        self.auto_last.setText(QCoreApplication.translate("Form", u"Auto Last", None))
    # retranslateUi

