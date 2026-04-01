# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'displayFrameUI.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFrame, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSpacerItem, QSplitter, QVBoxLayout,
    QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1440, 814)
        self.layout = QHBoxLayout(Form)
        self.layout.setSpacing(0)
        self.layout.setObjectName(u"layout")
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(Form)
        self.splitter.setObjectName(u"splitter")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy)
        self.splitter.setOrientation(Qt.Vertical)
        self.splitter.setHandleWidth(2)
        self.imageWindow = QFrame(self.splitter)
        self.imageWindow.setObjectName(u"imageWindow")
        sizePolicy.setHeightForWidth(self.imageWindow.sizePolicy().hasHeightForWidth())
        self.imageWindow.setSizePolicy(sizePolicy)
        self.imageWindow.setMinimumSize(QSize(0, 400))
        self.imageWindow.setFrameShape(QFrame.StyledPanel)
        self.imageWindow.setFrameShadow(QFrame.Raised)
        self.imageWindow.setLineWidth(3)
        self.verticalLayout_3 = QVBoxLayout(self.imageWindow)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.frame_top = QFrame(self.imageWindow)
        self.frame_top.setObjectName(u"frame_top")
        self.frame_top.setMinimumSize(QSize(0, 35))
        self.frame_top.setMaximumSize(QSize(16777215, 35))
        self.frame_top.setFrameShape(QFrame.StyledPanel)
        self.frame_top.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_5 = QHBoxLayout(self.frame_top)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.frame_4 = QFrame(self.frame_top)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setMinimumSize(QSize(260, 0))
        self.frame_4.setMaximumSize(QSize(260, 16777215))
        self.frame_4.setFrameShape(QFrame.NoFrame)
        self.frame_4.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_8 = QHBoxLayout(self.frame_4)
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(10, 0, 0, 0)
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setSpacing(0)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.normChannel = QComboBox(self.frame_4)
        self.normChannel.addItem("")
        self.normChannel.addItem("")
        self.normChannel.addItem("")
        self.normChannel.addItem("")
        self.normChannel.addItem("")
        self.normChannel.addItem("")
        self.normChannel.setObjectName(u"normChannel")
        self.normChannel.setMinimumSize(QSize(135, 0))
        self.normChannel.setMaximumSize(QSize(140, 16777215))
        self.normChannel.setFocusPolicy(Qt.StrongFocus)

        self.horizontalLayout_7.addWidget(self.normChannel)

        self.horizontalSpacer_3 = QSpacerItem(15, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_3)

        self.setBkg = QPushButton(self.frame_4)
        self.setBkg.setObjectName(u"setBkg")
        self.setBkg.setMinimumSize(QSize(90, 0))
        self.setBkg.setMaximumSize(QSize(100, 16777215))
        self.setBkg.setFocusPolicy(Qt.StrongFocus)

        self.horizontalLayout_7.addWidget(self.setBkg)

        self.horizontalSpacer_11 = QSpacerItem(10, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_11)


        self.horizontalLayout_8.addLayout(self.horizontalLayout_7)


        self.horizontalLayout_5.addWidget(self.frame_4)

        self.frame_5 = QFrame(self.frame_top)
        self.frame_5.setObjectName(u"frame_5")
        self.frame_5.setFrameShape(QFrame.NoFrame)
        self.frame_5.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.frame_5)
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.labelCurrent = QLabel(self.frame_5)
        self.labelCurrent.setObjectName(u"labelCurrent")
        self.labelCurrent.setMaximumSize(QSize(600, 16777215))
        font = QFont()
        font.setPointSize(15)
        self.labelCurrent.setFont(font)
        self.labelCurrent.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_6.addWidget(self.labelCurrent)


        self.horizontalLayout_5.addWidget(self.frame_5)

        self.frame_6 = QFrame(self.frame_top)
        self.frame_6.setObjectName(u"frame_6")
        self.frame_6.setMinimumSize(QSize(260, 0))
        self.frame_6.setMaximumSize(QSize(260, 16777215))
        self.frame_6.setFrameShape(QFrame.NoFrame)
        self.frame_6.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_10 = QHBoxLayout(self.frame_6)
        self.horizontalLayout_10.setSpacing(0)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.horizontalLayout_10.setContentsMargins(0, 0, 10, 0)
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setSpacing(0)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalSpacer_12 = QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_12)

        self.scale = QComboBox(self.frame_6)
        self.scale.addItem("")
        self.scale.addItem("")
        self.scale.addItem("")
        self.scale.addItem("")
        self.scale.setObjectName(u"scale")
        self.scale.setMinimumSize(QSize(80, 0))
        self.scale.setMaximumSize(QSize(80, 16777215))
        self.scale.setFocusPolicy(Qt.StrongFocus)

        self.horizontalLayout_9.addWidget(self.scale)

        self.horizontalSpacer_10 = QSpacerItem(15, 20, QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_10)

        self.cmap = QComboBox(self.frame_6)
        self.cmap.addItem("")
        self.cmap.addItem("")
        self.cmap.addItem("")
        self.cmap.addItem("")
        self.cmap.addItem("")
        self.cmap.addItem("")
        self.cmap.addItem("")
        self.cmap.addItem("")
        self.cmap.addItem("")
        self.cmap.addItem("")
        self.cmap.addItem("")
        self.cmap.setObjectName(u"cmap")
        self.cmap.setMinimumSize(QSize(80, 0))
        self.cmap.setMaximumSize(QSize(80, 16777215))
        self.cmap.setFocusPolicy(Qt.StrongFocus)

        self.horizontalLayout_9.addWidget(self.cmap)


        self.horizontalLayout_10.addLayout(self.horizontalLayout_9)


        self.horizontalLayout_5.addWidget(self.frame_6)


        self.verticalLayout_3.addWidget(self.frame_top)

        self.twoDWindow = QFrame(self.imageWindow)
        self.twoDWindow.setObjectName(u"twoDWindow")
        self.twoDWindow.setFrameShape(QFrame.StyledPanel)
        self.twoDWindow.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.twoDWindow)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.splitter_2 = QSplitter(self.twoDWindow)
        self.splitter_2.setObjectName(u"splitter_2")
        self.splitter_2.setOrientation(Qt.Horizontal)
        self.splitter_2.setHandleWidth(2)
        self.imageFrame = QFrame(self.splitter_2)
        self.imageFrame.setObjectName(u"imageFrame")
        self.imageFrame.setFrameShape(QFrame.StyledPanel)
        self.imageFrame.setFrameShadow(QFrame.Raised)
        self.splitter_2.addWidget(self.imageFrame)
        self.binnedFrame = QFrame(self.splitter_2)
        self.binnedFrame.setObjectName(u"binnedFrame")
        self.binnedFrame.setFrameShape(QFrame.StyledPanel)
        self.binnedFrame.setFrameShadow(QFrame.Raised)
        self.splitter_2.addWidget(self.binnedFrame)

        self.horizontalLayout_3.addWidget(self.splitter_2)


        self.verticalLayout_3.addWidget(self.twoDWindow)

        self.imageToolbar = QFrame(self.imageWindow)
        self.imageToolbar.setObjectName(u"imageToolbar")
        self.imageToolbar.setMinimumSize(QSize(0, 40))
        self.imageToolbar.setMaximumSize(QSize(16777215, 40))
        self.imageToolbar.setFrameShape(QFrame.NoFrame)
        self.imageToolbar.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.imageToolbar)
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(12, 0, -1, 0)
        self.imageUnit = QComboBox(self.imageToolbar)
        self.imageUnit.addItem("")
        self.imageUnit.addItem("")
        self.imageUnit.addItem("")
        self.imageUnit.setObjectName(u"imageUnit")
        self.imageUnit.setMinimumSize(QSize(90, 0))
        self.imageUnit.setMaximumSize(QSize(130, 16777215))
        self.imageUnit.setFocusPolicy(Qt.StrongFocus)

        self.horizontalLayout_2.addWidget(self.imageUnit)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.update2D = QCheckBox(self.imageToolbar)
        self.update2D.setObjectName(u"update2D")
        self.update2D.setFocusPolicy(Qt.StrongFocus)
        self.update2D.setChecked(True)

        self.horizontalLayout_2.addWidget(self.update2D)

        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_9)

        self.shareAxis = QCheckBox(self.imageToolbar)
        self.shareAxis.setObjectName(u"shareAxis")
        self.shareAxis.setMaximumSize(QSize(90, 16777215))
        self.shareAxis.setFocusPolicy(Qt.StrongFocus)

        self.horizontalLayout_2.addWidget(self.shareAxis)

        self.horizontalSpacer_14 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_14)

        self.save_2D = QPushButton(self.imageToolbar)
        self.save_2D.setObjectName(u"save_2D")

        self.horizontalLayout_2.addWidget(self.save_2D)


        self.verticalLayout_3.addWidget(self.imageToolbar)

        self.splitter.addWidget(self.imageWindow)
        self.plotWindow = QFrame(self.splitter)
        self.plotWindow.setObjectName(u"plotWindow")
        sizePolicy.setHeightForWidth(self.plotWindow.sizePolicy().hasHeightForWidth())
        self.plotWindow.setSizePolicy(sizePolicy)
        self.plotWindow.setMinimumSize(QSize(0, 400))
        self.plotWindow.setFrameShape(QFrame.StyledPanel)
        self.plotWindow.setFrameShadow(QFrame.Raised)
        self.plotWindow.setLineWidth(3)
        self.verticalLayout_4 = QVBoxLayout(self.plotWindow)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.oneDWindow = QFrame(self.plotWindow)
        self.oneDWindow.setObjectName(u"oneDWindow")
        self.oneDWindow.setFrameShape(QFrame.StyledPanel)
        self.oneDWindow.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.oneDWindow)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.splitter_3 = QSplitter(self.oneDWindow)
        self.splitter_3.setObjectName(u"splitter_3")
        self.splitter_3.setOrientation(Qt.Horizontal)
        self.splitter_3.setHandleWidth(2)
        self.plotFrame = QFrame(self.splitter_3)
        self.plotFrame.setObjectName(u"plotFrame")
        self.plotFrame.setFrameShape(QFrame.StyledPanel)
        self.plotFrame.setFrameShadow(QFrame.Raised)
        self.splitter_3.addWidget(self.plotFrame)

        self.horizontalLayout_4.addWidget(self.splitter_3)


        self.verticalLayout_4.addWidget(self.oneDWindow)

        self.plotToolBar = QFrame(self.plotWindow)
        self.plotToolBar.setObjectName(u"plotToolBar")
        self.plotToolBar.setMinimumSize(QSize(0, 40))
        self.plotToolBar.setMaximumSize(QSize(16777215, 40))
        self.plotToolBar.setFrameShape(QFrame.StyledPanel)
        self.plotToolBar.setFrameShadow(QFrame.Raised)
        self.horizontalLayout = QHBoxLayout(self.plotToolBar)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(12, 0, -1, 0)
        self.plotUnit = QComboBox(self.plotToolBar)
        self.plotUnit.addItem("")
        self.plotUnit.addItem("")
        self.plotUnit.addItem("")
        self.plotUnit.setObjectName(u"plotUnit")
        self.plotUnit.setMinimumSize(QSize(70, 0))
        self.plotUnit.setMaximumSize(QSize(100, 16777215))
        self.plotUnit.setFocusPolicy(Qt.StrongFocus)

        self.horizontalLayout.addWidget(self.plotUnit)

        self.slice = QCheckBox(self.plotToolBar)
        self.slice.setObjectName(u"slice")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.slice.sizePolicy().hasHeightForWidth())
        self.slice.setSizePolicy(sizePolicy1)
        self.slice.setMinimumSize(QSize(80, 0))
        self.slice.setMaximumSize(QSize(80, 16777215))

        self.horizontalLayout.addWidget(self.slice)

        self.slice_center = QDoubleSpinBox(self.plotToolBar)
        self.slice_center.setObjectName(u"slice_center")
        sizePolicy1.setHeightForWidth(self.slice_center.sizePolicy().hasHeightForWidth())
        self.slice_center.setSizePolicy(sizePolicy1)
        self.slice_center.setMinimumSize(QSize(70, 0))
        self.slice_center.setMaximumSize(QSize(70, 16777215))
        self.slice_center.setToolTipDuration(2)
        self.slice_center.setMinimum(-180.000000000000000)
        self.slice_center.setMaximum(180.000000000000000)
        self.slice_center.setSingleStep(0.500000000000000)
        self.slice_center.setValue(0.000000000000000)

        self.horizontalLayout.addWidget(self.slice_center)

        self.slice_width = QDoubleSpinBox(self.plotToolBar)
        self.slice_width.setObjectName(u"slice_width")
        sizePolicy1.setHeightForWidth(self.slice_width.sizePolicy().hasHeightForWidth())
        self.slice_width.setSizePolicy(sizePolicy1)
        self.slice_width.setMinimumSize(QSize(70, 0))
        self.slice_width.setMaximumSize(QSize(70, 16777215))
        self.slice_width.setMinimum(0.000000000000000)
        self.slice_width.setMaximum(270.000000000000000)
        self.slice_width.setSingleStep(0.500000000000000)
        self.slice_width.setValue(5.000000000000000)

        self.horizontalLayout.addWidget(self.slice_width)

        self.horizontalSpacer_4 = QSpacerItem(80, 20, QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_4)

        self.plotMethod = QComboBox(self.plotToolBar)
        self.plotMethod.addItem("")
        self.plotMethod.addItem("")
        self.plotMethod.addItem("")
        self.plotMethod.addItem("")
        self.plotMethod.addItem("")
        self.plotMethod.setObjectName(u"plotMethod")
        self.plotMethod.setFocusPolicy(Qt.StrongFocus)

        self.horizontalLayout.addWidget(self.plotMethod)

        self.yOffsetLabel = QLabel(self.plotToolBar)
        self.yOffsetLabel.setObjectName(u"yOffsetLabel")
        self.yOffsetLabel.setMaximumSize(QSize(140, 16777215))

        self.horizontalLayout.addWidget(self.yOffsetLabel)

        self.yOffset = QDoubleSpinBox(self.plotToolBar)
        self.yOffset.setObjectName(u"yOffset")
        self.yOffset.setEnabled(False)
        self.yOffset.setFocusPolicy(Qt.StrongFocus)
        self.yOffset.setDecimals(1)
        self.yOffset.setMinimum(-100.000000000000000)
        self.yOffset.setSingleStep(5.000000000000000)
        self.yOffset.setValue(5.000000000000000)

        self.horizontalLayout.addWidget(self.yOffset)

        self.wf_options = QPushButton(self.plotToolBar)
        self.wf_options.setObjectName(u"wf_options")
        self.wf_options.setEnabled(False)

        self.horizontalLayout.addWidget(self.wf_options)

        self.horizontalSpacer_6 = QSpacerItem(30, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_6)

        self.showLegend = QCheckBox(self.plotToolBar)
        self.showLegend.setObjectName(u"showLegend")
        self.showLegend.setFocusPolicy(Qt.StrongFocus)
        self.showLegend.setChecked(True)

        self.horizontalLayout.addWidget(self.showLegend)

        self.clear_1D = QPushButton(self.plotToolBar)
        self.clear_1D.setObjectName(u"clear_1D")

        self.horizontalLayout.addWidget(self.clear_1D)

        self.save_1D = QPushButton(self.plotToolBar)
        self.save_1D.setObjectName(u"save_1D")

        self.horizontalLayout.addWidget(self.save_1D)


        self.verticalLayout_4.addWidget(self.plotToolBar)

        self.splitter.addWidget(self.plotWindow)

        self.layout.addWidget(self.splitter)

        QWidget.setTabOrder(self.normChannel, self.setBkg)
        QWidget.setTabOrder(self.setBkg, self.scale)
        QWidget.setTabOrder(self.scale, self.cmap)
        QWidget.setTabOrder(self.cmap, self.imageUnit)
        QWidget.setTabOrder(self.imageUnit, self.save_2D)
        QWidget.setTabOrder(self.save_2D, self.plotUnit)
        QWidget.setTabOrder(self.plotUnit, self.slice_center)
        QWidget.setTabOrder(self.slice_center, self.slice_width)
        QWidget.setTabOrder(self.slice_width, self.plotMethod)
        QWidget.setTabOrder(self.plotMethod, self.yOffset)
        QWidget.setTabOrder(self.yOffset, self.save_1D)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.normChannel.setItemText(0, QCoreApplication.translate("Form", u"Norm Channel", None))
        self.normChannel.setItemText(1, QCoreApplication.translate("Form", u"Monitor", None))
        self.normChannel.setItemText(2, QCoreApplication.translate("Form", u"sec", None))
        self.normChannel.setItemText(3, QCoreApplication.translate("Form", u"bstop", None))
        self.normChannel.setItemText(4, QCoreApplication.translate("Form", u"I0", None))
        self.normChannel.setItemText(5, QCoreApplication.translate("Form", u"I1", None))

        self.setBkg.setText(QCoreApplication.translate("Form", u"Set Bkg", None))
        self.labelCurrent.setText(QCoreApplication.translate("Form", u"Current", None))
        self.scale.setItemText(0, QCoreApplication.translate("Form", u"Linear", None))
        self.scale.setItemText(1, QCoreApplication.translate("Form", u"Log", None))
        self.scale.setItemText(2, QCoreApplication.translate("Form", u"Log-Log", None))
        self.scale.setItemText(3, QCoreApplication.translate("Form", u"Sqrt", None))

        self.cmap.setItemText(0, QCoreApplication.translate("Form", u"Default", None))
        self.cmap.setItemText(1, QCoreApplication.translate("Form", u"viridis", None))
        self.cmap.setItemText(2, QCoreApplication.translate("Form", u"grey", None))
        self.cmap.setItemText(3, QCoreApplication.translate("Form", u"plasma", None))
        self.cmap.setItemText(4, QCoreApplication.translate("Form", u"inferno", None))
        self.cmap.setItemText(5, QCoreApplication.translate("Form", u"magma", None))
        self.cmap.setItemText(6, QCoreApplication.translate("Form", u"thermal", None))
        self.cmap.setItemText(7, QCoreApplication.translate("Form", u"flame", None))
        self.cmap.setItemText(8, QCoreApplication.translate("Form", u"yellowy", None))
        self.cmap.setItemText(9, QCoreApplication.translate("Form", u"bipolar", None))
        self.cmap.setItemText(10, QCoreApplication.translate("Form", u"greyclip", None))

        self.imageUnit.setItemText(0, QCoreApplication.translate("Form", u"Q-Chi", None))
        self.imageUnit.setItemText(1, QCoreApplication.translate("Form", u"2Th-Chi", None))
        self.imageUnit.setItemText(2, QCoreApplication.translate("Form", u"Qz-Qxy", None))

        self.update2D.setText(QCoreApplication.translate("Form", u"Update 2D", None))
        self.shareAxis.setText(QCoreApplication.translate("Form", u"Share Axis", None))
        self.save_2D.setText(QCoreApplication.translate("Form", u"Save", None))
        self.plotUnit.setItemText(0, QCoreApplication.translate("Form", u"Q (A-1)", None))
        self.plotUnit.setItemText(1, QCoreApplication.translate("Form", u"2 u\\u03B8", None))
        self.plotUnit.setItemText(2, QCoreApplication.translate("Form", u"Chi", None))

        self.slice.setText(QCoreApplication.translate("Form", u"X Range", None))
#if QT_CONFIG(tooltip)
        self.slice_center.setToolTip("")
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        self.slice_center.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(accessibility)
        self.slice_center.setAccessibleDescription("")
#endif // QT_CONFIG(accessibility)
#if QT_CONFIG(tooltip)
        self.slice_width.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.plotMethod.setItemText(0, QCoreApplication.translate("Form", u"Single", None))
        self.plotMethod.setItemText(1, QCoreApplication.translate("Form", u"Overlay", None))
        self.plotMethod.setItemText(2, QCoreApplication.translate("Form", u"Average", None))
        self.plotMethod.setItemText(3, QCoreApplication.translate("Form", u"Sum", None))
        self.plotMethod.setItemText(4, QCoreApplication.translate("Form", u"Waterfall", None))

#if QT_CONFIG(tooltip)
        self.yOffsetLabel.setToolTip(QCoreApplication.translate("Form", u"y Offset for Overlay Mode", None))
#endif // QT_CONFIG(tooltip)
        self.yOffsetLabel.setText(QCoreApplication.translate("Form", u"Offset", None))
#if QT_CONFIG(tooltip)
        self.wf_options.setToolTip("")
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        self.wf_options.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
        self.wf_options.setText(QCoreApplication.translate("Form", u"Options", None))
        self.showLegend.setText(QCoreApplication.translate("Form", u"Legend", None))
        self.clear_1D.setText(QCoreApplication.translate("Form", u"Clear", None))
        self.save_1D.setText(QCoreApplication.translate("Form", u"Save", None))
    # retranslateUi

