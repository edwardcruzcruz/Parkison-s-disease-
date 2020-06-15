from PyQt5 import QtCore, QtGui, QtWidgets
import bcrypt
import sqlite3
import sys
from enum import Enum
import os
import argparse
import sqlite3
import datetime
import subprocess

from MainWindow import *

class User_type(Enum):
    ADMIN = 1
    NORMAL = 2

class Admin_windows(Enum):
    PRINCIPAL = 1
    NETWORK = 2

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def is_user(user, password):
    passwordToCompare = str.encode(password)
    conn = None
    try:
        conn = sqlite3.connect("theDatabase.db")
    except Error as e:
        print(e)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * from USER
     """)
    rows = cursor.fetchall()
    for row in rows:
        userToCompare = row[1]
        hashed_pass = row[2]
        user_permission = row[3]
        if user == userToCompare and bcrypt.checkpw(passwordToCompare, hashed_pass):
            return True, user_permission
    return False, None

def sql_connection():
    try:
        con = sqlite3.connect('theDatabase.db')
        return con
    except Error:
        print(Error)

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
        self.user_edit.setObjectName("user_edit")
        self.edits_layout.addWidget(self.user_edit)
        self.password_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.password_edit.setFont(font)
        self.password_edit.setObjectName("password_edit")
        self.password_edit.setEchoMode(QtWidgets.QLineEdit.Password)
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

class Login(QtWidgets.QMainWindow, Ui_LoginWindow):

    switch_window = QtCore.pyqtSignal(int)

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setupUi(self)

        self.signIn_button.clicked.connect(self.login_functionality)

    def login_functionality(self):
        the_user = self.user_edit.text()
        the_pass = self.password_edit.text()
        exist_user, user_permission = is_user(the_user, the_pass)
        if exist_user:
            if (user_permission == User_type.ADMIN.value):
                self.switch_window.emit(User_type.ADMIN.value)
            else: 
                self.switch_window.emit(User_type.NORMAL.value)
        else:
            msg = QtWidgets.QMessageBox(self.centralwidget)
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Login or password incorrect!')
            msg.setWindowTitle("Error")
            msg.exec_()

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
        self.images_button.clicked.connect(self.new_images_functionality)
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
        self.model_button.clicked.connect(self.model_functionality)
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

    def new_images_functionality(self):
        msg = QtWidgets.QMessageBox.information(self, 'Nii images', "Select all the Nii images for training. Images MUST have an X size of 38 or 63.", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Cancel)
        if msg == QtWidgets.QMessageBox.Ok:
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            files, _ = QtWidgets.QFileDialog.getOpenFileNames(self,"Nii images", "","Nii Files (*.nii.gz)", options=options)
            if files:
                msg2 = QtWidgets.QMessageBox.information(self, 'Mask images', "Select all the MASK Nii images for training. Images MUST have an X size of 38 or 63. Also must be selected in the same order as the nii images.", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Cancel)
                if msg2 == QtWidgets.QMessageBox.Ok:
                    options2 = QtWidgets.QFileDialog.Options()
                    options2 |= QtWidgets.QFileDialog.DontUseNativeDialog
                    files2, _ = QtWidgets.QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","Nii Files (*.nii.gz)", options=options2)
                    if files2:
                        if len(files) == len(files2):
                            tupleList = []
                            conn = sql_connection()
                            cursorObj = conn.cursor()
                            sqlite_insert = """INSERT INTO 'image'
                                ('name', 'pathImage', 'pathMask') 
                                VALUES (?, ?, ? );"""
                            for imageNumber in range(0, len(files)):
                                nameList = files[imageNumber].split(os.sep)
                                name = nameList[len(nameList) - 1]
                                tupleInsert = (name, files[imageNumber], files2[imageNumber])
                                tupleList.append(tupleInsert)
                            cursorObj.executemany(sqlite_insert, tupleList)
                            conn.commit()
                            conn.close()
                            QtWidgets.QMessageBox.information(self, 'New images', "New images add correctly, new model will be trained in background", QtWidgets.QMessageBox.Ok)
                            value_os = "python2 " if os.name == "nt" else "python "
                            python3_command = value_os + "python2/run_new_model.py"  # launch your python2 script using bash
                            process = subprocess.Popen(python3_command.split(), stdout=subprocess.PIPE)
                            output, error = process.communicate()  # receive output from the python2 script

                        else:
                            error3 = QtWidgets.QMessageBox.critical(self, 'Critical', "Different number of masks selected!! ", QtWidgets.QMessageBox.Ok)
                    else:
                        error2 = QtWidgets.QMessageBox.critical(self, 'Not files selected', "Not Nii files selected for new masks", QtWidgets.QMessageBox.Ok)
            else:
                error1 = QtWidgets.QMessageBox.critical(self, 'Not files selected', "Not Nii files selected for new images", QtWidgets.QMessageBox.Ok)

    
class Admin(QtWidgets.QMainWindow, Ui_AdminWindow):

    switch_window = QtCore.pyqtSignal(int)

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setupUi(self)

        self.goto_button.clicked.connect(self.goto_functionality)

    def goto_functionality(self):
        self.switch_window.emit(Admin_windows.PRINCIPAL.value)
    
    def model_functionality(self):
        self.switch_window.emit(Admin_windows.NETWORK.value)
    
class Ui_NewNetworkWindow(object):
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
        self.onlyInt = QtGui.QIntValidator()
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
        self.z_edit.setValidator(self.onlyInt)
        self.z_edit.setGeometry(QtCore.QRect(100, 160, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.z_edit.setFont(font)
        self.z_edit.setObjectName("z_edit")
        self.y_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.y_edit.setValidator(self.onlyInt)
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


class NewNetworkWindow(QtWidgets.QMainWindow, Ui_NewNetworkWindow):

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setupUi(self)

        self.pushButton.clicked.connect(self.train_new_model)

    def train_new_model(self):
        msg = QtWidgets.QMessageBox.information(self, 'New Model', "Your new model with this new parameters will be trained on background. It will use all current images.", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Cancel)
        conn = sql_connection()
        cursorObj = conn.cursor()
        creationDate = datetime.datetime.now()
        executionDate = datetime.datetime.now()
        sqlite_insert = """INSERT INTO 'model'
                        ('name', 'path', 'Y', 'Z','creationDate', 'executionDate') 
                        VALUES (?, ?, ?, ?, ?, ?);"""
        tuple_insert = (self.name_edit.text(),'',self.y_edit.text(),self.z_edit.text(), creationDate, executionDate )
        cursorObj.execute(sqlite_insert, tuple_insert)
        conn.commit()
        conn.close()
        self.close()
        value_os = "python2 " if os.name == "nt" else "python "
        python3_command = value_os + "python2/run_new_model.py"  # launch your python2 script using bash
        process = subprocess.Popen(python3_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()  # receive output from the python2 script
        

class Controller:

    def __init__(self, app):
        self.app = app
        pass

    def show_login(self):
        self.login = Login()
        self.login.switch_window.connect(self.show_depending)
        self.login.show()

    def show_depending(self,parameter):
        if parameter == User_type.ADMIN.value:
            """
            self.admin = Admin()
            self.admin.switch_window.connect(self.show_main)
            self.login.close()
            self.admin.show()
            """
            redirect_vtk_messages()
            self.app.BRAIN_FILE = 'data/100/orig/FLAIR.nii.gz'
            self.app.MASK_FILE = 'data/100/wmh.nii.gz'
            self.window = MainWindow(self.app, User_type.ADMIN.value)
            self.login.close()
            self.window.show()
        else:
            redirect_vtk_messages()
            self.app.BRAIN_FILE = 'data/100/orig/FLAIR.nii.gz'
            self.app.MASK_FILE = 'data/100/wmh.nii.gz'
            self.window = MainWindow(self.app, User_type.NORMAL.value)
            self.login.close()
            self.window.show()

    
    def show_main(self, parameter):
        if parameter == Admin_windows.NETWORK.value:
            self.new_network = NewNetworkWindow()
            self.new_network.show()
        else:
            redirect_vtk_messages()
            self.app.BRAIN_FILE = 'data/100/orig/FLAIR.nii.gz'
            self.app.MASK_FILE = 'data/100/wmh.nii.gz'
            self.window = MainWindow(self.app)
            self.admin.close()
            self.window.show()

def redirect_vtk_messages():
    """ Redirect VTK related error messages to a file."""
    import tempfile
    tempfile.template = 'vtk-err'
    f = tempfile.mktemp('.log')
    log = vtk.vtkFileOutputWindow()
    log.SetFlush(1)
    log.SetFileName(f)
    log.SetInstance(log)

def verify_type(file):
    ext = os.path.basename(file).split(os.extsep, 1)
    if ext[1] != 'nii.gz':
        parser.error("File doesn't end with 'nii.gz'. Found: {}".format(ext[1]))
    return file

def main():
    app = QtWidgets.QApplication(sys.argv)
    controller = Controller(app)
    controller.show_login()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
