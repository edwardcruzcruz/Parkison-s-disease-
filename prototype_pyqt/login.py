# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\login.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_LoginWindow(object):
    def setupUi(self, LoginWindow):
        LoginWindow.setObjectName("LoginWindow")
        LoginWindow.resize(329, 317)
        self.centralwidget = QtWidgets.QWidget(LoginWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.signIn_button = QtWidgets.QPushButton(self.centralwidget)
        self.signIn_button.setGeometry(QtCore.QRect(90, 220, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.signIn_button.setFont(font)
        self.signIn_button.setAutoDefault(False)
        self.signIn_button.setDefault(False)
        self.signIn_button.setFlat(False)
        self.signIn_button.setObjectName("signIn_button")
        self.logo = QtWidgets.QLabel(self.centralwidget)
        self.logo.setGeometry(QtCore.QRect(110, 10, 91, 91))
        self.logo.setText("")
        self.logo.setPixmap(QtGui.QPixmap("resources/logotype.png"))
        self.logo.setScaledContents(True)
        self.logo.setObjectName("logo")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(160, 110, 141, 101))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.edits_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.edits_layout.setContentsMargins(0, 0, 0, 0)
        self.edits_layout.setObjectName("edits_layout")
        self.user_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.user_edit.setFont(font)
        self.user_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.user_edit.setObjectName("user_edit")
        self.edits_layout.addWidget(self.user_edit)
        self.password_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.password_edit.setFont(font)
        self.password_edit.setObjectName("password_edit")
        self.edits_layout.addWidget(self.password_edit)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(20, 110, 141, 101))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.label_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.label_layout.setContentsMargins(0, 0, 0, 0)
        self.label_layout.setObjectName("label_layout")
        self.user_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.user_label.setFont(font)
        self.user_label.setObjectName("user_label")
        self.label_layout.addWidget(self.user_label)
        self.password_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.password_label.setFont(font)
        self.password_label.setObjectName("password_label")
        self.label_layout.addWidget(self.password_label)
        LoginWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(LoginWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 329, 26))
        self.menubar.setObjectName("menubar")
        LoginWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(LoginWindow)
        self.statusbar.setObjectName("statusbar")
        LoginWindow.setStatusBar(self.statusbar)

        self.retranslateUi(LoginWindow)
        QtCore.QMetaObject.connectSlotsByName(LoginWindow)

    def retranslateUi(self, LoginWindow):
        _translate = QtCore.QCoreApplication.translate
        LoginWindow.setWindowTitle(_translate("LoginWindow", "MainWindow"))
        self.signIn_button.setText(_translate("LoginWindow", "Sign in"))
        self.user_label.setText(_translate("LoginWindow", "User:"))
        self.password_label.setText(_translate("LoginWindow", "Password:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    LoginWindow = QtWidgets.QMainWindow()
    ui = Ui_LoginWindow()
    ui.setupUi(LoginWindow)
    LoginWindow.show()
    sys.exit(app.exec_())
