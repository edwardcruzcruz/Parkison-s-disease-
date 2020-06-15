# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\new_network.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_NewModel(object):
    def setupUi(self, NewModel):
        NewModel.setObjectName("NewModel")
        NewModel.resize(289, 347)
        self.centralwidget = QtWidgets.QWidget(NewModel)
        self.centralwidget.setObjectName("centralwidget")
        self.name_label = QtWidgets.QLabel(self.centralwidget)
        self.name_label.setGeometry(QtCore.QRect(30, 40, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.name_label.setFont(font)
        self.name_label.setObjectName("name_label")
        self.z_label = QtWidgets.QLabel(self.centralwidget)
        self.z_label.setGeometry(QtCore.QRect(30, 160, 21, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.z_label.setFont(font)
        self.z_label.setObjectName("z_label")
        self.y_label = QtWidgets.QLabel(self.centralwidget)
        self.y_label.setGeometry(QtCore.QRect(30, 100, 21, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.y_label.setFont(font)
        self.y_label.setObjectName("y_label")
        self.name_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.name_edit.setGeometry(QtCore.QRect(100, 40, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.name_edit.setFont(font)
        self.name_edit.setObjectName("name_edit")
        self.z_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.z_edit.setGeometry(QtCore.QRect(100, 160, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.z_edit.setFont(font)
        self.z_edit.setObjectName("z_edit")
        self.y_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.y_edit.setGeometry(QtCore.QRect(100, 100, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.y_edit.setFont(font)
        self.y_edit.setObjectName("y_edit")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(80, 230, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        NewModel.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(NewModel)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 289, 26))
        self.menubar.setObjectName("menubar")
        NewModel.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(NewModel)
        self.statusbar.setObjectName("statusbar")
        NewModel.setStatusBar(self.statusbar)

        self.retranslateUi(NewModel)
        QtCore.QMetaObject.connectSlotsByName(NewModel)

    def retranslateUi(self, NewModel):
        _translate = QtCore.QCoreApplication.translate
        NewModel.setWindowTitle(_translate("NewModel", "MainWindow"))
        self.name_label.setText(_translate("NewModel", "Name:"))
        self.z_label.setText(_translate("NewModel", "Z:"))
        self.y_label.setText(_translate("NewModel", "Y:"))
        self.pushButton.setText(_translate("NewModel", "OK"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    NewModel = QtWidgets.QMainWindow()
    ui = Ui_NewModel()
    ui.setupUi(NewModel)
    NewModel.show()
    sys.exit(app.exec_())
