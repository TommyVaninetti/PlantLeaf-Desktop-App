# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mathDialog2.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
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
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QFrame,
    QHBoxLayout, QLabel, QLayout, QLineEdit,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget)

class Ui_QDialogMath(object):
    def setupUi(self, QDialogMath):
        if not QDialogMath.objectName():
            QDialogMath.setObjectName(u"QDialogMath")
        QDialogMath.resize(1200, 800)
        self.horizontalLayout_3 = QHBoxLayout(QDialogMath)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.totalHLayout = QHBoxLayout()
        self.totalHLayout.setObjectName(u"totalHLayout")
        self.plotSectionVLayout = QVBoxLayout()
        self.plotSectionVLayout.setObjectName(u"plotSectionVLayout")
        self.plotWidget = QWidget(QDialogMath)
        self.plotWidget.setObjectName(u"plotWidget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plotWidget.sizePolicy().hasHeightForWidth())
        self.plotWidget.setSizePolicy(sizePolicy)

        self.plotSectionVLayout.addWidget(self.plotWidget)

        self.formulaSeparator = QFrame(QDialogMath)
        self.formulaSeparator.setObjectName(u"formulaSeparator")
        self.formulaSeparator.setFrameShape(QFrame.Shape.HLine)
        self.formulaSeparator.setFrameShadow(QFrame.Shadow.Raised)

        self.plotSectionVLayout.addWidget(self.formulaSeparator)

        self.formulaWidget = QLabel(QDialogMath)
        self.formulaWidget.setObjectName(u"formulaWidget")

        self.plotSectionVLayout.addWidget(self.formulaWidget)

        self.notesAndButtonsHLayout = QHBoxLayout()
        self.notesAndButtonsHLayout.setObjectName(u"notesAndButtonsHLayout")
        self.notesLineEdit = QLineEdit(QDialogMath)
        self.notesLineEdit.setObjectName(u"notesLineEdit")

        self.notesAndButtonsHLayout.addWidget(self.notesLineEdit)

        self.upperButtonsHLayout = QHBoxLayout()
        self.upperButtonsHLayout.setObjectName(u"upperButtonsHLayout")
        self.autoFitButton = QPushButton(QDialogMath)
        self.autoFitButton.setObjectName(u"autoFitButton")

        self.upperButtonsHLayout.addWidget(self.autoFitButton)

        self.moreInfoButton = QPushButton(QDialogMath)
        self.moreInfoButton.setObjectName(u"moreInfoButton")

        self.upperButtonsHLayout.addWidget(self.moreInfoButton)

        self.analysisLinesButton = QPushButton(QDialogMath)
        self.analysisLinesButton.setObjectName(u"analysisLinesButton")

        self.upperButtonsHLayout.addWidget(self.analysisLinesButton)

        self.exportButton = QPushButton(QDialogMath)
        self.exportButton.setObjectName(u"exportButton")

        self.upperButtonsHLayout.addWidget(self.exportButton)


        self.notesAndButtonsHLayout.addLayout(self.upperButtonsHLayout)


        self.plotSectionVLayout.addLayout(self.notesAndButtonsHLayout)


        self.totalHLayout.addLayout(self.plotSectionVLayout)

        self.separatorFrame = QFrame(QDialogMath)
        self.separatorFrame.setObjectName(u"separatorFrame")
        self.separatorFrame.setFrameShape(QFrame.Shape.VLine)
        self.separatorFrame.setFrameShadow(QFrame.Shadow.Plain)

        self.totalHLayout.addWidget(self.separatorFrame)

        self.resultsSectionVLayout = QVBoxLayout()
        self.resultsSectionVLayout.setObjectName(u"resultsSectionVLayout")
        self.resultsSectionVLayout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        self.infoPrefaceVLayout = QVBoxLayout()
        self.infoPrefaceVLayout.setObjectName(u"infoPrefaceVLayout")
        self.titleLabel = QLabel(QDialogMath)
        self.titleLabel.setObjectName(u"titleLabel")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.titleLabel.sizePolicy().hasHeightForWidth())
        self.titleLabel.setSizePolicy(sizePolicy1)

        self.infoPrefaceVLayout.addWidget(self.titleLabel, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)

        self.fitSelectionLayout = QHBoxLayout()
        self.fitSelectionLayout.setObjectName(u"fitSelectionLayout")
        self.signalTypeComboBox = QComboBox(QDialogMath)
        self.signalTypeComboBox.setObjectName(u"signalTypeComboBox")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.signalTypeComboBox.sizePolicy().hasHeightForWidth())
        self.signalTypeComboBox.setSizePolicy(sizePolicy2)
        self.signalTypeComboBox.setMaximumSize(QSize(16777215, 30))

        self.fitSelectionLayout.addWidget(self.signalTypeComboBox)


        self.infoPrefaceVLayout.addLayout(self.fitSelectionLayout)


        self.resultsSectionVLayout.addLayout(self.infoPrefaceVLayout)

        self.resultsWidget = QWidget(QDialogMath)
        self.resultsWidget.setObjectName(u"resultsWidget")
        sizePolicy.setHeightForWidth(self.resultsWidget.sizePolicy().hasHeightForWidth())
        self.resultsWidget.setSizePolicy(sizePolicy)

        self.resultsSectionVLayout.addWidget(self.resultsWidget)

        self.buttonsVLayout = QVBoxLayout()
        self.buttonsVLayout.setObjectName(u"buttonsVLayout")
        self.downButtonsHLayout = QHBoxLayout()
        self.downButtonsHLayout.setObjectName(u"downButtonsHLayout")
        self.saveButton = QPushButton(QDialogMath)
        self.saveButton.setObjectName(u"saveButton")

        self.downButtonsHLayout.addWidget(self.saveButton)

        self.cancelButton = QPushButton(QDialogMath)
        self.cancelButton.setObjectName(u"cancelButton")

        self.downButtonsHLayout.addWidget(self.cancelButton)


        self.buttonsVLayout.addLayout(self.downButtonsHLayout)


        self.resultsSectionVLayout.addLayout(self.buttonsVLayout)


        self.totalHLayout.addLayout(self.resultsSectionVLayout)


        self.horizontalLayout_3.addLayout(self.totalHLayout)


        self.retranslateUi(QDialogMath)

        QMetaObject.connectSlotsByName(QDialogMath)
    # setupUi

    def retranslateUi(self, QDialogMath):
        QDialogMath.setWindowTitle(QCoreApplication.translate("QDialogMath", u"QDialogMath", None))
        self.formulaWidget.setText("")
        self.autoFitButton.setText(QCoreApplication.translate("QDialogMath", u"AutoFit", None))
        self.moreInfoButton.setText(QCoreApplication.translate("QDialogMath", u"More Info", None))
        self.analysisLinesButton.setText(QCoreApplication.translate("QDialogMath", u"Hide Lines", None))
        self.exportButton.setText(QCoreApplication.translate("QDialogMath", u"Export as CSV", None))
        self.titleLabel.setText(QCoreApplication.translate("QDialogMath", u"Analysis", None))
        self.saveButton.setText(QCoreApplication.translate("QDialogMath", u"Save", None))
        self.cancelButton.setText(QCoreApplication.translate("QDialogMath", u"Cancel", None))
    # retranslateUi

