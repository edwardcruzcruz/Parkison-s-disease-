# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\admin.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_AdminWindow(object):
    def setupUi(self, AdminWindow):
        AdminWindow.setObjectName("AdminWindow")
        AdminWindow.resize(336, 451)
        self.centralwidget = QtWidgets.QWidget(AdminWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.images_widget = QtWidgets.QFrame(self.centralwidget)
        self.images_widget.setGeometry(QtCore.QRect(30, 30, 261, 111))
        self.images_widget.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.images_widget.setFrameShadow(QtWidgets.QFrame.Raised)
        self.images_widget.setObjectName("images_widget")
        self.images_button = QtWidgets.QPushButton(self.images_widget)
        self.images_button.setGeometry(QtCore.QRect(90, 10, 73, 69))
        self.images_button.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("resources/plus.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.images_button.setIcon(icon)
        self.images_button.setIconSize(QtCore.QSize(60, 60))
        self.images_button.setObjectName("images_button")
        self.images_label = QtWidgets.QLabel(self.images_widget)
        self.images_label.setGeometry(QtCore.QRect(10, 90, 251, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.images_label.setFont(font)
        self.images_label.setObjectName("images_label")
        self.model_widget = QtWidgets.QFrame(self.centralwidget)
        self.model_widget.setGeometry(QtCore.QRect(30, 160, 261, 111))
        self.model_widget.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.model_widget.setFrameShadow(QtWidgets.QFrame.Raised)
        self.model_widget.setObjectName("model_widget")
        self.model_button = QtWidgets.QPushButton(self.model_widget)
        self.model_button.setGeometry(QtCore.QRect(90, 10, 73, 69))
        self.model_button.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("resources/neural_network.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.model_button.setIcon(icon1)
        self.model_button.setIconSize(QtCore.QSize(60, 60))
        self.model_button.setObjectName("model_button")
        self.model_label = QtWidgets.QLabel(self.model_widget)
        self.model_label.setGeometry(QtCore.QRect(20, 90, 231, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.model_label.setFont(font)
        self.model_label.setObjectName("model_label")
        self.goto_widget = QtWidgets.QFrame(self.centralwidget)
        self.goto_widget.setGeometry(QtCore.QRect(30, 290, 261, 111))
        self.goto_widget.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.goto_widget.setFrameShadow(QtWidgets.QFrame.Raised)
        self.goto_widget.setObjectName("goto_widget")
        self.goto_button = QtWidgets.QPushButton(self.goto_widget)
        self.goto_button.setGeometry(QtCore.QRect(90, 10, 73, 69))
        self.goto_button.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("resources/right_arrow.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.goto_button.setIcon(icon2)
        self.goto_button.setIconSize(QtCore.QSize(60, 60))
        self.goto_button.setObjectName("goto_button")
        self.goto_label = QtWidgets.QLabel(self.goto_widget)
        self.goto_label.setGeometry(QtCore.QRect(60, 90, 141, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.goto_label.setFont(font)
        self.goto_label.setObjectName("goto_label")
        AdminWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(AdminWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 336, 26))
        self.menubar.setObjectName("menubar")
        AdminWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(AdminWindow)
        self.statusbar.setObjectName("statusbar")
        AdminWindow.setStatusBar(self.statusbar)

        self.retranslateUi(AdminWindow)
        QtCore.QMetaObject.connectSlotsByName(AdminWindow)

    def retranslateUi(self, AdminWindow):
        _translate = QtCore.QCoreApplication.translate
        AdminWindow.setWindowTitle(_translate("AdminWindow", "MainWindow"))
        self.images_label.setText(_translate("AdminWindow", "Add new images to actual model"))
        self.model_label.setText(_translate("AdminWindow", "Add new deep learning model"))
        self.goto_label.setText(_translate("AdminWindow", "Go to visualization"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    AdminWindow = QtWidgets.QMainWindow()
    ui = Ui_AdminWindow()
    ui.setupUi(AdminWindow)
    AdminWindow.show()
    sys.exit(app.exec_())
