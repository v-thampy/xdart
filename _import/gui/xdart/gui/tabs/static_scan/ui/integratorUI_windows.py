# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'integratorUI_windows.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QSpacerItem, QVBoxLayout,
    QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1440, 260)
        Form.setMaximumSize(QSize(16777215, 280))
        self.verticalLayout = QVBoxLayout(Form)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.frame1D = QFrame(Form)
        self.frame1D.setObjectName(u"frame1D")
        self.frame1D.setMaximumSize(QSize(16777215, 120))
        self.frame1D.setFrameShape(QFrame.StyledPanel)
        self.frame1D.setFrameShadow(QFrame.Raised)
        self.frame1D.setLineWidth(20)
        self.frame1D.setMidLineWidth(5)
        self.horizontalLayout = QHBoxLayout(self.frame1D)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.frame1D_layout = QVBoxLayout()
        self.frame1D_layout.setSpacing(0)
        self.frame1D_layout.setObjectName(u"frame1D_layout")
        self.frame1D_header = QFrame(self.frame1D)
        self.frame1D_header.setObjectName(u"frame1D_header")
        self.frame1D_header.setMaximumSize(QSize(16777215, 50))
        self.frame1D_header.setAutoFillBackground(True)
        self.frame1D_header.setStyleSheet(u"")
        self.frame1D_header.setFrameShape(QFrame.NoFrame)
        self.frame1D_header.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.frame1D_header)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(15, 3, 5, 3)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label1D = QLabel(self.frame1D_header)
        self.label1D.setObjectName(u"label1D")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label1D.sizePolicy().hasHeightForWidth())
        self.label1D.setSizePolicy(sizePolicy)
        self.label1D.setMinimumSize(QSize(50, 0))
        self.label1D.setMaximumSize(QSize(50, 16777215))
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        self.label1D.setFont(font)
        self.label1D.setAutoFillBackground(False)
        self.label1D.setStyleSheet(u"background-color: rgba(195, 195, 195, 200);")
        self.label1D.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_3.addWidget(self.label1D)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer)

        self.axis1D = QComboBox(self.frame1D_header)
        self.axis1D.addItem("")
        self.axis1D.setObjectName(u"axis1D")

        self.horizontalLayout_3.addWidget(self.axis1D)

        self.horizontalSpacer_1 = QSpacerItem(30, 20, QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_1)

        self.label_npts_1D = QLabel(self.frame1D_header)
        self.label_npts_1D.setObjectName(u"label_npts_1D")
        self.label_npts_1D.setMaximumSize(QSize(40, 16777215))
        self.label_npts_1D.setIndent(-1)

        self.horizontalLayout_3.addWidget(self.label_npts_1D)

        self.npts_1D = QLineEdit(self.frame1D_header)
        self.npts_1D.setObjectName(u"npts_1D")
        self.npts_1D.setMaximumSize(QSize(55, 16777215))
        self.npts_1D.setInputMethodHints(Qt.ImhDigitsOnly)

        self.horizontalLayout_3.addWidget(self.npts_1D)


        self.horizontalLayout_4.addLayout(self.horizontalLayout_3)


        self.frame1D_layout.addWidget(self.frame1D_header)

        self.frame1D_range = QFrame(self.frame1D)
        self.frame1D_range.setObjectName(u"frame1D_range")
        self.frame1D_range.setMaximumSize(QSize(16777215, 71))
        self.frame1D_range.setFrameShape(QFrame.NoFrame)
        self.frame1D_range.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.frame1D_range)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(5, 5, 5, 5)
        self.gridLayout_1D = QGridLayout()
        self.gridLayout_1D.setObjectName(u"gridLayout_1D")
        self.gridLayout_1D.setHorizontalSpacing(6)
        self.radial_autoRange_1D = QCheckBox(self.frame1D_range)
        self.radial_autoRange_1D.setObjectName(u"radial_autoRange_1D")
        self.radial_autoRange_1D.setChecked(True)

        self.gridLayout_1D.addWidget(self.radial_autoRange_1D, 0, 4, 1, 1)

        self.unit_1D = QComboBox(self.frame1D_range)
        self.unit_1D.addItem("")
        self.unit_1D.addItem("")
        self.unit_1D.setObjectName(u"unit_1D")
        self.unit_1D.setMaximumSize(QSize(90, 16777215))

        self.gridLayout_1D.addWidget(self.unit_1D, 0, 0, 1, 1)

        self.label_to1 = QLabel(self.frame1D_range)
        self.label_to1.setObjectName(u"label_to1")
        self.label_to1.setMaximumSize(QSize(100, 30))

        self.gridLayout_1D.addWidget(self.label_to1, 1, 2, 1, 1)

        self.azim_low_1D = QLineEdit(self.frame1D_range)
        self.azim_low_1D.setObjectName(u"azim_low_1D")

        self.gridLayout_1D.addWidget(self.azim_low_1D, 1, 1, 1, 1)

        self.radial_low_1D = QLineEdit(self.frame1D_range)
        self.radial_low_1D.setObjectName(u"radial_low_1D")

        self.gridLayout_1D.addWidget(self.radial_low_1D, 0, 1, 1, 1)

        self.azim_autoRange_1D = QCheckBox(self.frame1D_range)
        self.azim_autoRange_1D.setObjectName(u"azim_autoRange_1D")
        self.azim_autoRange_1D.setChecked(True)

        self.gridLayout_1D.addWidget(self.azim_autoRange_1D, 1, 4, 1, 1)

        self.azim_high_1D = QLineEdit(self.frame1D_range)
        self.azim_high_1D.setObjectName(u"azim_high_1D")

        self.gridLayout_1D.addWidget(self.azim_high_1D, 1, 3, 1, 1)

        self.label_to2 = QLabel(self.frame1D_range)
        self.label_to2.setObjectName(u"label_to2")

        self.gridLayout_1D.addWidget(self.label_to2, 0, 2, 1, 1)

        self.label_azim_1D = QLabel(self.frame1D_range)
        self.label_azim_1D.setObjectName(u"label_azim_1D")
        self.label_azim_1D.setMinimumSize(QSize(40, 0))
        self.label_azim_1D.setMaximumSize(QSize(40, 50))
        self.label_azim_1D.setAlignment(Qt.AlignCenter)

        self.gridLayout_1D.addWidget(self.label_azim_1D, 1, 0, 1, 1)

        self.radial_high_1D = QLineEdit(self.frame1D_range)
        self.radial_high_1D.setObjectName(u"radial_high_1D")

        self.gridLayout_1D.addWidget(self.radial_high_1D, 0, 3, 1, 1)


        self.horizontalLayout_2.addLayout(self.gridLayout_1D)


        self.frame1D_layout.addWidget(self.frame1D_range)

        # Button frame hidden — Re-Integrate/Advanced moved elsewhere.
        # Widgets kept as hidden stubs so existing signal connections don't break.
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        font1 = QFont()
        font1.setFamilies([u"MS Shell Dlg 2"])
        font1.setPointSize(9)
        self.frame1D_buttons = QFrame(self.frame1D)
        self.frame1D_buttons.setObjectName(u"frame1D_buttons")
        self.frame1D_buttons.setMaximumSize(QSize(0, 0))
        self.frame1D_buttons.setVisible(False)
        self.integrate1D = QPushButton(self.frame1D_buttons)
        self.integrate1D.setObjectName(u"integrate1D")
        self.integrate1D.setVisible(False)
        self.advanced1D = QPushButton(self.frame1D_buttons)
        self.advanced1D.setObjectName(u"advanced1D")
        self.advanced1D.setVisible(False)

        self.frame1D_layout.addWidget(self.frame1D_buttons)


        self.horizontalLayout.addLayout(self.frame1D_layout)


        self.verticalLayout.addWidget(self.frame1D)

        self.frame2D = QFrame(Form)
        self.frame2D.setObjectName(u"frame2D")
        self.frame2D.setMaximumSize(QSize(16777215, 120))
        self.frame2D.setFrameShape(QFrame.StyledPanel)
        self.frame2D.setFrameShadow(QFrame.Raised)
        self.frame2D.setLineWidth(20)
        self.frame2D.setMidLineWidth(5)
        self.horizontalLayout_12 = QHBoxLayout(self.frame2D)
        self.horizontalLayout_12.setSpacing(0)
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.horizontalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.frame2D_layout = QVBoxLayout()
        self.frame2D_layout.setSpacing(0)
        self.frame2D_layout.setObjectName(u"frame2D_layout")
        self.frame2D_header = QFrame(self.frame2D)
        self.frame2D_header.setObjectName(u"frame2D_header")
        self.frame2D_header.setMaximumSize(QSize(16777215, 50))
        self.frame2D_header.setAutoFillBackground(True)
        self.frame2D_header.setFrameShape(QFrame.NoFrame)
        self.frame2D_header.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_7 = QHBoxLayout(self.frame2D_header)
        self.horizontalLayout_7.setSpacing(0)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(15, 3, 5, 3)
        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.label2D = QLabel(self.frame2D_header)
        self.label2D.setObjectName(u"label2D")
        sizePolicy.setHeightForWidth(self.label2D.sizePolicy().hasHeightForWidth())
        self.label2D.setSizePolicy(sizePolicy)
        self.label2D.setMinimumSize(QSize(50, 0))
        self.label2D.setMaximumSize(QSize(50, 16777215))
        self.label2D.setFont(font)
        self.label2D.setAutoFillBackground(False)
        self.label2D.setStyleSheet(u"background-color: rgba(195, 195, 195, 200);")
        self.label2D.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_8.addWidget(self.label2D)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_8.addItem(self.horizontalSpacer_2)

        self.axis2D = QComboBox(self.frame2D_header)
        self.axis2D.addItem("")
        self.axis2D.addItem("")
        self.axis2D.setObjectName(u"axis2D")

        self.horizontalLayout_8.addWidget(self.axis2D)

        self.horizontalSpacer_3 = QSpacerItem(30, 20, QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_8.addItem(self.horizontalSpacer_3)

        self.label_npts_2D = QLabel(self.frame2D_header)
        self.label_npts_2D.setObjectName(u"label_npts_2D")
        self.label_npts_2D.setMaximumSize(QSize(40, 16777215))
        self.label_npts_2D.setIndent(-1)

        self.horizontalLayout_8.addWidget(self.label_npts_2D)

        self.npts_radial_2D = QLineEdit(self.frame2D_header)
        self.npts_radial_2D.setObjectName(u"npts_radial_2D")
        self.npts_radial_2D.setMaximumSize(QSize(55, 16777215))
        self.npts_radial_2D.setInputMethodHints(Qt.ImhDigitsOnly)

        self.horizontalLayout_8.addWidget(self.npts_radial_2D)

        self.npts_azim_2D = QLineEdit(self.frame2D_header)
        self.npts_azim_2D.setObjectName(u"npts_azim_2D")
        self.npts_azim_2D.setMaximumSize(QSize(55, 16777215))
        self.npts_azim_2D.setInputMethodHints(Qt.ImhDigitsOnly)

        self.horizontalLayout_8.addWidget(self.npts_azim_2D)


        self.horizontalLayout_7.addLayout(self.horizontalLayout_8)


        self.frame2D_layout.addWidget(self.frame2D_header)

        self.frame2D_range = QFrame(self.frame2D)
        self.frame2D_range.setObjectName(u"frame2D_range")
        self.frame2D_range.setMaximumSize(QSize(16777215, 71))
        self.frame2D_range.setFrameShape(QFrame.NoFrame)
        self.frame2D_range.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_9 = QHBoxLayout(self.frame2D_range)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(5, 5, 5, 5)
        self.gridLayout_2D = QGridLayout()
        self.gridLayout_2D.setObjectName(u"gridLayout_2D")
        self.gridLayout_2D.setHorizontalSpacing(6)
        self.radial_autoRange_2D = QCheckBox(self.frame2D_range)
        self.radial_autoRange_2D.setObjectName(u"radial_autoRange_2D")
        self.radial_autoRange_2D.setChecked(True)

        self.gridLayout_2D.addWidget(self.radial_autoRange_2D, 0, 4, 1, 1)

        self.unit_2D = QComboBox(self.frame2D_range)
        self.unit_2D.addItem("")
        self.unit_2D.addItem("")
        self.unit_2D.setObjectName(u"unit_2D")
        self.unit_2D.setMaximumSize(QSize(90, 16777215))

        self.gridLayout_2D.addWidget(self.unit_2D, 0, 0, 1, 1)

        self.label_to1_2 = QLabel(self.frame2D_range)
        self.label_to1_2.setObjectName(u"label_to1_2")
        self.label_to1_2.setMaximumSize(QSize(100, 30))

        self.gridLayout_2D.addWidget(self.label_to1_2, 1, 2, 1, 1)

        self.azim_low_2D = QLineEdit(self.frame2D_range)
        self.azim_low_2D.setObjectName(u"azim_low_2D")

        self.gridLayout_2D.addWidget(self.azim_low_2D, 1, 1, 1, 1)

        self.radial_low_2D = QLineEdit(self.frame2D_range)
        self.radial_low_2D.setObjectName(u"radial_low_2D")

        self.gridLayout_2D.addWidget(self.radial_low_2D, 0, 1, 1, 1)

        self.azim_autoRange_2D = QCheckBox(self.frame2D_range)
        self.azim_autoRange_2D.setObjectName(u"azim_autoRange_2D")
        self.azim_autoRange_2D.setChecked(True)

        self.gridLayout_2D.addWidget(self.azim_autoRange_2D, 1, 4, 1, 1)

        self.azim_high_2D = QLineEdit(self.frame2D_range)
        self.azim_high_2D.setObjectName(u"azim_high_2D")

        self.gridLayout_2D.addWidget(self.azim_high_2D, 1, 3, 1, 1)

        self.label_to2_2 = QLabel(self.frame2D_range)
        self.label_to2_2.setObjectName(u"label_to2_2")

        self.gridLayout_2D.addWidget(self.label_to2_2, 0, 2, 1, 1)

        self.label_azim_2D = QLabel(self.frame2D_range)
        self.label_azim_2D.setObjectName(u"label_azim_2D")
        self.label_azim_2D.setMinimumSize(QSize(40, 0))
        self.label_azim_2D.setMaximumSize(QSize(40, 50))
        self.label_azim_2D.setAlignment(Qt.AlignCenter)

        self.gridLayout_2D.addWidget(self.label_azim_2D, 1, 0, 1, 1)

        self.radial_high_2D = QLineEdit(self.frame2D_range)
        self.radial_high_2D.setObjectName(u"radial_high_2D")

        self.gridLayout_2D.addWidget(self.radial_high_2D, 0, 3, 1, 1)


        self.horizontalLayout_9.addLayout(self.gridLayout_2D)


        self.frame2D_layout.addWidget(self.frame2D_range)

        # Button frame hidden — Re-Integrate/Advanced moved elsewhere.
        # Widgets kept as hidden stubs so existing signal connections don't break.
        self.frame2D_buttons = QFrame(self.frame2D)
        self.frame2D_buttons.setObjectName(u"frame2D_buttons")
        self.frame2D_buttons.setMaximumSize(QSize(0, 0))
        self.frame2D_buttons.setVisible(False)
        self.integrate2D = QPushButton(self.frame2D_buttons)
        self.integrate2D.setObjectName(u"integrate2D")
        self.integrate2D.setVisible(False)
        self.advanced2D = QPushButton(self.frame2D_buttons)
        self.advanced2D.setObjectName(u"advanced2D")
        self.advanced2D.setVisible(False)

        self.frame2D_layout.addWidget(self.frame2D_buttons)


        self.horizontalLayout_12.addLayout(self.frame2D_layout)


        self.verticalLayout.addWidget(self.frame2D)

        self.frame_3 = QFrame(Form)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setMaximumSize(QSize(16777215, 50))
        self.frame_3.setAutoFillBackground(True)
        self.frame_3.setStyleSheet(u"")
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_14 = QHBoxLayout(self.frame_3)
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.horizontalLayout_14.setContentsMargins(5, 0, 5, 0)
        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setSpacing(6)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.raw_to_tif = QPushButton(self.frame_3)
        self.raw_to_tif.setObjectName(u"raw_to_tif")
        font2 = QFont()
        font2.setFamilies([u"MS Shell Dlg 2"])
        font2.setPointSize(10)
        self.raw_to_tif.setFont(font2)

        self.horizontalLayout_13.addWidget(self.raw_to_tif)

        self.pyfai_calib = QPushButton(self.frame_3)
        self.pyfai_calib.setObjectName(u"pyfai_calib")
        self.pyfai_calib.setMinimumSize(QSize(0, 20))
        self.pyfai_calib.setFont(font2)

        self.horizontalLayout_13.addWidget(self.pyfai_calib)

        self.get_mask = QPushButton(self.frame_3)
        self.get_mask.setObjectName(u"get_mask")
        self.get_mask.setFont(font2)

        self.horizontalLayout_13.addWidget(self.get_mask)


        self.horizontalLayout_14.addLayout(self.horizontalLayout_13)


        self.verticalLayout.addWidget(self.frame_3)

#if QT_CONFIG(shortcut)
        self.label_npts_1D.setBuddy(self.npts_1D)
        self.label_npts_2D.setBuddy(self.npts_radial_2D)
#endif // QT_CONFIG(shortcut)
        QWidget.setTabOrder(self.axis1D, self.npts_1D)
        QWidget.setTabOrder(self.npts_1D, self.unit_1D)
        QWidget.setTabOrder(self.unit_1D, self.radial_low_1D)
        QWidget.setTabOrder(self.radial_low_1D, self.radial_high_1D)
        QWidget.setTabOrder(self.radial_high_1D, self.radial_autoRange_1D)
        QWidget.setTabOrder(self.radial_autoRange_1D, self.azim_low_1D)
        QWidget.setTabOrder(self.azim_low_1D, self.azim_high_1D)
        QWidget.setTabOrder(self.azim_high_1D, self.azim_autoRange_1D)
        QWidget.setTabOrder(self.azim_autoRange_1D, self.integrate1D)
        QWidget.setTabOrder(self.integrate1D, self.advanced1D)
        QWidget.setTabOrder(self.advanced1D, self.axis2D)
        QWidget.setTabOrder(self.axis2D, self.npts_radial_2D)
        QWidget.setTabOrder(self.npts_radial_2D, self.npts_azim_2D)
        QWidget.setTabOrder(self.npts_azim_2D, self.unit_2D)
        QWidget.setTabOrder(self.unit_2D, self.radial_low_2D)
        QWidget.setTabOrder(self.radial_low_2D, self.radial_high_2D)
        QWidget.setTabOrder(self.radial_high_2D, self.radial_autoRange_2D)
        QWidget.setTabOrder(self.radial_autoRange_2D, self.azim_low_2D)
        QWidget.setTabOrder(self.azim_low_2D, self.azim_high_2D)
        QWidget.setTabOrder(self.azim_high_2D, self.azim_autoRange_2D)
        QWidget.setTabOrder(self.azim_autoRange_2D, self.integrate2D)
        QWidget.setTabOrder(self.integrate2D, self.advanced2D)
        QWidget.setTabOrder(self.advanced2D, self.pyfai_calib)
        QWidget.setTabOrder(self.pyfai_calib, self.get_mask)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.label1D.setText(QCoreApplication.translate("Form", u"1-D", None))
        self.axis1D.setItemText(0, QCoreApplication.translate("Form", u"Radial", None))

        self.label_npts_1D.setText(QCoreApplication.translate("Form", u"Points", None))
        self.npts_1D.setText(QCoreApplication.translate("Form", u"3000", None))
        self.npts_1D.setPlaceholderText(QCoreApplication.translate("Form", u"500", None))
        self.radial_autoRange_1D.setText(QCoreApplication.translate("Form", u"Auto", None))
        self.unit_1D.setItemText(0, QCoreApplication.translate("Form", u"q (u\\u212Bu\\u207Bu\\u00B9)", None))
        self.unit_1D.setItemText(1, QCoreApplication.translate("Form", u"2 u\\u03B8", None))

        self.label_to1.setText(QCoreApplication.translate("Form", u"to", None))
        self.azim_low_1D.setText(QCoreApplication.translate("Form", u"-180", None))
        self.radial_low_1D.setText(QCoreApplication.translate("Form", u"0", None))
        self.azim_autoRange_1D.setText(QCoreApplication.translate("Form", u"Auto", None))
        self.azim_high_1D.setText(QCoreApplication.translate("Form", u"180", None))
        self.label_to2.setText(QCoreApplication.translate("Form", u"to", None))
        self.label_azim_1D.setText(QCoreApplication.translate("Form", u"Chi", None))
        self.radial_high_1D.setText(QCoreApplication.translate("Form", u"5", None))
        self.integrate1D.setText(QCoreApplication.translate("Form", u"Re-Integrate", None))
        self.advanced1D.setText(QCoreApplication.translate("Form", u"Advanced...", None))
        self.label2D.setText(QCoreApplication.translate("Form", u"2-D", None))
        self.axis2D.setItemText(0, QCoreApplication.translate("Form", u"Q-Chi", None))
        self.axis2D.setItemText(1, QCoreApplication.translate("Form", u"Qz-Qxy", None))

        self.label_npts_2D.setText(QCoreApplication.translate("Form", u"Points", None))
        self.npts_radial_2D.setText(QCoreApplication.translate("Form", u"500", None))
        self.npts_radial_2D.setPlaceholderText(QCoreApplication.translate("Form", u"500", None))
        self.npts_azim_2D.setText(QCoreApplication.translate("Form", u"500", None))
        self.radial_autoRange_2D.setText(QCoreApplication.translate("Form", u"Auto", None))
        self.unit_2D.setItemText(0, QCoreApplication.translate("Form", u"q (u\\u212Bu\\u207Bu\\u00B9)", None))
        self.unit_2D.setItemText(1, QCoreApplication.translate("Form", u"2 u\\u03B8", None))

        self.label_to1_2.setText(QCoreApplication.translate("Form", u"to", None))
        self.azim_low_2D.setText(QCoreApplication.translate("Form", u"-180", None))
        self.radial_low_2D.setText(QCoreApplication.translate("Form", u"0", None))
        self.azim_autoRange_2D.setText(QCoreApplication.translate("Form", u"Auto", None))
        self.azim_high_2D.setText(QCoreApplication.translate("Form", u"180", None))
        self.label_to2_2.setText(QCoreApplication.translate("Form", u"to", None))
        self.label_azim_2D.setText(QCoreApplication.translate("Form", u"Chi", None))
        self.radial_high_2D.setText(QCoreApplication.translate("Form", u"5", None))
        self.integrate2D.setText(QCoreApplication.translate("Form", u"Re-Integrate", None))
        self.advanced2D.setText(QCoreApplication.translate("Form", u"Advanced...", None))
        self.raw_to_tif.setText(QCoreApplication.translate("Form", u"Raw -> Tif", None))
        self.pyfai_calib.setText(QCoreApplication.translate("Form", u"Calibrate", None))
        self.get_mask.setText(QCoreApplication.translate("Form", u"Make Mask", None))
    # retranslateUi

