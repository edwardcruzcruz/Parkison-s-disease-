import math
import time
import os

import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkUtils import *
from config import *
from enum import Enum
import sqlite3
import subprocess
import datetime
import numpy as np
import SimpleITK as sitk
import dicom2nifti
import threading
import time
from pathlib import Path

from os import listdir
from os.path import isfile, join
class User_type(Enum):
    ADMIN = 1
    NORMAL = 2

def sql_connection():
    try:
        con = sqlite3.connect('theDatabase.db')
        return con
    except Error:
        print(Error)

class MainWindow(QtWidgets.QMainWindow, QtWidgets.QApplication):
    def __init__(self, app, mode):
        self.app = app
        self.mode = mode
        QtWidgets.QMainWindow.__init__(self, None)

        # base setup
        self.renderer, self.frame, self.vtk_widget, self.interactor, self.render_window = self.setup()
        self.brain, self.mask = setup_brain(self.renderer, self.app.BRAIN_FILE), setup_mask(self.renderer,
                                                                                            self.app.MASK_FILE)

        #menu bar
        mainMenu = self.menuBar()     
        fileMenu = mainMenu.addMenu('New Segmentation...')
        exitAct = QtWidgets.QAction('&Add MRI images...', self)        
        exitAct.setShortcut('Ctrl+M')
        exitAct.setStatusTip('Add image')
        exitAct.triggered.connect(self.openFileNameDialog)
        self.statusBar()

        fileMenu.addAction(exitAct)

        # setup brain projection and slicer
        self.brain_image_prop = setup_projection(self.brain, self.renderer)
        self.brain_slicer_props = setup_slicer(self.renderer, self.brain)  # causing issues with rotation
        self.slicer_widgets = []

        # brain pickers
        self.brain_threshold_sp = self.create_new_picker(self.brain.scalar_range[1], self.brain.scalar_range[0], 5.0,
                                                         sum(self.brain.scalar_range) / 2, self.brain_threshold_vc)
        self.brain_opacity_sp = self.create_new_picker(1.0, 0.0, 0.1, BRAIN_OPACITY, self.brain_opacity_vc)
        self.brain_smoothness_sp = self.create_new_picker(1000, 100, 100, BRAIN_SMOOTHNESS, self.brain_smoothness_vc)
        self.brain_lut_sp = self.create_new_picker(3.0, 0.0, 0.1, 2.0, self.lut_value_changed)
        self.brain_projection_cb = self.add_brain_projection()
        self.brain_slicer_cb = self.add_brain_slicer()

        # mask pickers
        self.mask_opacity_sp = self.create_new_picker(1.0, 0.0, 0.1, MASK_OPACITY, self.mask_opacity_vc)
        self.mask_smoothness_sp = self.create_new_picker(1000, 100, 100, MASK_SMOOTHNESS, self.mask_smoothness_vc)
        self.mask_label_cbs = []

        # create grid for all widgets
        self.grid = QtWidgets.QGridLayout()

        # add each widget
        self.add_vtk_window_widget()
        self.add_brain_settings_widget()
        self.add_mask_settings_widget()
        self.add_views_widget()
        if self.mode == User_type.ADMIN.value:
            self.add_admin_widget()

        #  set layout and show
        self.render_window.Render()
        self.setWindowTitle(APPLICATION_TITLE)
        self.frame.setLayout(self.grid)
        self.setCentralWidget(self.frame)
        self.set_axial_view()
        self.interactor.Initialize()
        self.show()

    @staticmethod
    def setup():
        """
        Create and setup the base vtk and Qt objects for the application
        """
        renderer = vtk.vtkRenderer()
        frame = QtWidgets.QFrame()
        vtk_widget = QVTKRenderWindowInteractor()
        interactor = vtk_widget.GetRenderWindow().GetInteractor()
        render_window = vtk_widget.GetRenderWindow()

        frame.setAutoFillBackground(True)
        vtk_widget.GetRenderWindow().AddRenderer(renderer)
        render_window.AddRenderer(renderer)
        interactor.SetRenderWindow(render_window)
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        # required to enable overlapping actors with opacity < 1.0
        # this is causing some issues with flashing objects
        # render_window.SetAlphaBitPlanes(1)
        # render_window.SetMultiSamples(0)
        # renderer.UseDepthPeelingOn()
        # renderer.SetMaximumNumberOfPeels(2)

        return renderer, frame, vtk_widget, interactor, render_window

    def lut_value_changed(self):
        lut = self.brain.image_mapper.GetLookupTable()
        new_lut_value = self.brain_lut_sp.value()
        lut.SetValueRange(0.0, new_lut_value)
        lut.Build()
        self.brain.image_mapper.SetLookupTable(lut)
        self.brain.image_mapper.Update()
        self.render_window.Render()

    def add_brain_slicer(self):
        slicer_cb = QtWidgets.QCheckBox("Slicer")
        slicer_cb.clicked.connect(self.brain_slicer_vc)
        return slicer_cb

    def add_vtk_window_widget(self):
        base_brain_file = os.path.basename(self.app.BRAIN_FILE)
        base_mask_file = os.path.basename(self.app.MASK_FILE)
        object_title = "Brain: {0} (min: {1:.2f}, max: {2:.2f})        Mask: {3}".format(base_brain_file,
                                                                                         self.brain.scalar_range[0],
                                                                                         self.brain.scalar_range[1],
                                                                                         base_mask_file)
        object_group_box = QtWidgets.QGroupBox(object_title)
        object_layout = QtWidgets.QVBoxLayout()
        object_layout.addWidget(self.vtk_widget)
        object_group_box.setLayout(object_layout)
        if self.mode == User_type.ADMIN.value:
            self.grid.addWidget(object_group_box, 0, 2, 7, 5)
        else:
            self.grid.addWidget(object_group_box, 0, 2, 5, 5)
        # must manually set column width for vtk_widget to maintain height:width ratio
        self.grid.setColumnMinimumWidth(2, 700)

    def add_brain_settings_widget(self):
        brain_group_box = QtWidgets.QGroupBox("Brain Settings")
        brain_group_layout = QtWidgets.QGridLayout()
        brain_group_layout.addWidget(QtWidgets.QLabel("Brain Threshold"), 0, 0)
        brain_group_layout.addWidget(QtWidgets.QLabel("Brain Opacity"), 1, 0)
        brain_group_layout.addWidget(QtWidgets.QLabel("Brain Smoothness"), 2, 0)
        brain_group_layout.addWidget(QtWidgets.QLabel("Image Intensity"), 3, 0)
        brain_group_layout.addWidget(self.brain_threshold_sp, 0, 1, 1, 2)
        brain_group_layout.addWidget(self.brain_opacity_sp, 1, 1, 1, 2)
        brain_group_layout.addWidget(self.brain_smoothness_sp, 2, 1, 1, 2)
        brain_group_layout.addWidget(self.brain_lut_sp, 3, 1, 1, 2)
        brain_group_layout.addWidget(self.brain_projection_cb, 4, 0)
        brain_group_layout.addWidget(self.brain_slicer_cb, 4, 1)
        brain_group_layout.addWidget(self.create_new_separator(), 5, 0, 1, 3)
        brain_group_layout.addWidget(QtWidgets.QLabel("Axial Slice"), 6, 0)
        brain_group_layout.addWidget(QtWidgets.QLabel("Coronal Slice"), 7, 0)
        brain_group_layout.addWidget(QtWidgets.QLabel("Sagittal Slice"), 8, 0)

        # order is important
        slicer_funcs = [self.axial_slice_changed, self.coronal_slice_changed, self.sagittal_slice_changed]
        current_label_row = 6
        # data extent is array [xmin, xmax, ymin, ymax, zmin, zmax)
        # we want all the max values for the range
        extent_index = 5
        for func in slicer_funcs:
            slice_widget = QtWidgets.QSlider(Qt.Qt.Horizontal)
            slice_widget.setDisabled(True)
            self.slicer_widgets.append(slice_widget)
            brain_group_layout.addWidget(slice_widget, current_label_row, 1, 1, 2)
            slice_widget.valueChanged.connect(func)
            slice_widget.setRange(self.brain.extent[extent_index - 1], self.brain.extent[extent_index])
            slice_widget.setValue(self.brain.extent[extent_index] / 2)
            current_label_row += 1
            extent_index -= 2

        brain_group_box.setLayout(brain_group_layout)
        self.grid.addWidget(brain_group_box, 0, 0, 1, 2)

    def axial_slice_changed(self):
        pos = self.slicer_widgets[0].value()
        self.brain_slicer_props[0].SetDisplayExtent(self.brain.extent[0], self.brain.extent[1], self.brain.extent[2],
                                                    self.brain.extent[3], pos, pos)
        self.render_window.Render()

    def coronal_slice_changed(self):
        pos = self.slicer_widgets[1].value()
        self.brain_slicer_props[1].SetDisplayExtent(self.brain.extent[0], self.brain.extent[1], pos, pos,
                                                    self.brain.extent[4], self.brain.extent[5])
        self.render_window.Render()

    def sagittal_slice_changed(self):
        pos = self.slicer_widgets[2].value()
        self.brain_slicer_props[2].SetDisplayExtent(pos, pos, self.brain.extent[2], self.brain.extent[3],
                                                    self.brain.extent[4], self.brain.extent[5])
        self.render_window.Render()

    def add_mask_settings_widget(self):
        mask_settings_group_box = QtWidgets.QGroupBox("Mask Settings")
        mask_settings_layout = QtWidgets.QGridLayout()
        mask_settings_layout.addWidget(QtWidgets.QLabel("Mask Opacity"), 0, 0)
        mask_settings_layout.addWidget(QtWidgets.QLabel("Mask Smoothness"), 1, 0)
        mask_settings_layout.addWidget(self.mask_opacity_sp, 0, 1)
        mask_settings_layout.addWidget(self.mask_smoothness_sp, 1, 1)
        mask_multi_color_radio = QtWidgets.QRadioButton("Multi Color")
        mask_multi_color_radio.setChecked(True)
        mask_multi_color_radio.clicked.connect(self.mask_multi_color_radio_checked)
        mask_single_color_radio = QtWidgets.QRadioButton("Single Color")
        mask_single_color_radio.clicked.connect(self.mask_single_color_radio_checked)
        mask_settings_layout.addWidget(mask_multi_color_radio, 2, 0)
        mask_settings_layout.addWidget(mask_single_color_radio, 2, 1)
        mask_settings_layout.addWidget(self.create_new_separator(), 3, 0, 1, 2)

        self.mask_label_cbs = []
        c_col, c_row = 0, 4  # c_row must always be (+1) of last row
        for i in range(1, 11):
            self.mask_label_cbs.append(QtWidgets.QCheckBox("Label {}".format(i)))
            mask_settings_layout.addWidget(self.mask_label_cbs[i - 1], c_row, c_col)
            c_row = c_row + 1 if c_col == 1 else c_row
            c_col = 0 if c_col == 1 else 1

        mask_settings_group_box.setLayout(mask_settings_layout)
        self.grid.addWidget(mask_settings_group_box, 1, 0, 2, 2)

        for i, cb in enumerate(self.mask_label_cbs):
            if i < len(self.mask.labels) and self.mask.labels[i].actor:
                cb.setChecked(True)
                cb.clicked.connect(self.mask_label_checked)
            else:
                cb.setDisabled(True)

    def add_views_widget(self):
        axial_view = QtWidgets.QPushButton("Axial")
        coronal_view = QtWidgets.QPushButton("Coronal")
        sagittal_view = QtWidgets.QPushButton("Sagittal")
        views_box = QtWidgets.QGroupBox("Views")
        views_box_layout = QtWidgets.QVBoxLayout()
        views_box_layout.addWidget(axial_view)
        views_box_layout.addWidget(coronal_view)
        views_box_layout.addWidget(sagittal_view)
        views_box.setLayout(views_box_layout)
        self.grid.addWidget(views_box, 3, 0, 2, 2)
        axial_view.clicked.connect(self.set_axial_view)
        coronal_view.clicked.connect(self.set_coronal_view)
        sagittal_view.clicked.connect(self.set_sagittal_view)

    def add_admin_widget(self):
        new_model_view = QtWidgets.QPushButton("New model")
        new_images_view = QtWidgets.QPushButton("New images")
        admin_box = QtWidgets.QGroupBox("Admin")
        admin_box_layout = QtWidgets.QVBoxLayout()
        admin_box_layout.addWidget(new_model_view)
        admin_box_layout.addWidget(new_images_view)
        admin_box.setLayout(admin_box_layout)
        self.grid.addWidget(admin_box, 5, 0, 2, 2)
        new_model_view.clicked.connect(self.new_model_functionality)
        new_images_view.clicked.connect(self.new_images_functionality)

    @staticmethod
    def create_new_picker(max_value, min_value, step, picker_value, value_changed_func):
        if isinstance(max_value, int):
            picker = QtWidgets.QSpinBox()
        else:
            picker = QtWidgets.QDoubleSpinBox()

        picker.setMaximum(max_value)
        picker.setMinimum(min_value)
        picker.setSingleStep(step)
        picker.setValue(picker_value)
        picker.valueChanged.connect(value_changed_func)
        return picker

    def add_brain_projection(self):
        projection_cb = QtWidgets.QCheckBox("Projection")
        projection_cb.clicked.connect(self.brain_projection_vc)
        return projection_cb

    def mask_label_checked(self):
        for i, cb in enumerate(self.mask_label_cbs):
            if cb.isChecked():
                self.mask.labels[i].property.SetOpacity(self.mask_opacity_sp.value())
            elif cb.isEnabled():  # labels without data are disabled
                self.mask.labels[i].property.SetOpacity(0)
        self.render_window.Render()

    def mask_single_color_radio_checked(self):
        for label in self.mask.labels:
            if label.property:
                label.property.SetColor(MASK_COLORS[0])
        self.render_window.Render()

    def mask_multi_color_radio_checked(self):
        for label in self.mask.labels:
            if label.property:
                label.property.SetColor(label.color)
        self.render_window.Render()

    def brain_projection_vc(self):
        projection_checked = self.brain_projection_cb.isChecked()
        self.brain_slicer_cb.setDisabled(projection_checked)  # disable slicer checkbox, cant use both at same time
        self.brain_image_prop.SetOpacity(projection_checked)
        self.render_window.Render()

    def brain_slicer_vc(self):
        slicer_checked = self.brain_slicer_cb.isChecked()

        for widget in self.slicer_widgets:
            widget.setEnabled(slicer_checked)

        self.brain_projection_cb.setDisabled(slicer_checked)  # disable projection checkbox, cant use both at same time
        for prop in self.brain_slicer_props:
            prop.GetProperty().SetOpacity(slicer_checked)
        self.render_window.Render()

    def brain_opacity_vc(self):
        opacity = round(self.brain_opacity_sp.value(), 2)
        self.brain.labels[0].property.SetOpacity(opacity)
        self.render_window.Render()

    def brain_threshold_vc(self):
        self.process_changes()
        threshold = self.brain_threshold_sp.value()
        self.brain.labels[0].extractor.SetValue(0, threshold)
        self.render_window.Render()

    def brain_smoothness_vc(self):
        self.process_changes()
        smoothness = self.brain_smoothness_sp.value()
        self.brain.labels[0].smoother.SetNumberOfIterations(smoothness)
        self.render_window.Render()

    def mask_opacity_vc(self):
        opacity = round(self.mask_opacity_sp.value(), 2)
        for i, label in enumerate(self.mask.labels):
            if label.property and self.mask_label_cbs[i].isChecked():
                label.property.SetOpacity(opacity)
        self.render_window.Render()

    def mask_smoothness_vc(self):
        self.process_changes()
        smoothness = self.mask_smoothness_sp.value()
        for label in self.mask.labels:
            if label.smoother:
                label.smoother.SetNumberOfIterations(smoothness)
        self.render_window.Render()

    def creating_files(self,fileName):
        if not os.path.isdir('temporal'):
            os.makedirs('temporal')
        a = dicom2nifti.convert_directory(fileName, "temporal")
    
    def update_image(self,brain,mask):

        self.app.BRAIN_FILE = brain
        self.app.MASK_FILE = mask

        self.window = MainWindow(self.app, self.mode)

        self.close()

        # base setup

    def openFileNameDialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName =  QtWidgets.QFileDialog.getExistingDirectory(None, 'Select a folder:', 'C:\\', QtWidgets.QFileDialog.ShowDirsOnly)
        print(fileName)
        if fileName:
            print("Creating nifti files from..." + fileName)
            #self.dataset = pydicom.dcmread(fileName)
            #print(self.dataset)
            """
            try:
                subprocess.Popen(["python file_creator.py " + fileName + "temporal"])
            except:
                pass
            """
            dicom2nifti.convert_directory(fileName, 'temporal')
            for f in listdir("temporal"):
                if "flair" in f:
                    print('file creator finished')
                    ruta_flair = join("temporal/", f)
                    name_flair_1 = ruta_flair.split(os.sep)
                    name_flair_2 = name_flair_1[len(name_flair_1) -1].split('.')[0] + '_mask.nii.gz'
                    print(ruta_flair)
                    print(name_flair_2)
                    value_os = "python2 " if os.name == "nt" else "python "
                    python3_command = value_os + "python2/generate_wmh_mask.py " + ruta_flair + ' ' + name_flair_2 # launch your python2 script using bash
                    process = subprocess.Popen(python3_command.split(), stdout=subprocess.PIPE)
                    counter = 0
                    while True:
                        if Path(name_flair_2).is_file():
                            print('im inside here')
                            self.update_image(ruta_flair,name_flair_2)
                            break
                        time.sleep(1)
                        counter +=1
                        if counter > 12:
                            break


    def set_axial_view(self):
        self.renderer.ResetCamera()
        fp = self.renderer.GetActiveCamera().GetFocalPoint()
        p = self.renderer.GetActiveCamera().GetPosition()
        dist = math.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1]) ** 2 + (p[2] - fp[2]) ** 2)
        self.renderer.GetActiveCamera().SetPosition(fp[0], fp[1], fp[2] + dist)
        self.renderer.GetActiveCamera().SetViewUp(0.0, 1.0, 0.0)
        self.renderer.GetActiveCamera().Zoom(1.8)
        self.render_window.Render()

    def set_coronal_view(self):
        self.renderer.ResetCamera()
        fp = self.renderer.GetActiveCamera().GetFocalPoint()
        p = self.renderer.GetActiveCamera().GetPosition()
        dist = math.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1]) ** 2 + (p[2] - fp[2]) ** 2)
        self.renderer.GetActiveCamera().SetPosition(fp[0], fp[2] - dist, fp[1])
        self.renderer.GetActiveCamera().SetViewUp(0.0, 0.5, 0.5)
        self.renderer.GetActiveCamera().Zoom(1.8)
        self.render_window.Render()

    def set_sagittal_view(self):
        self.renderer.ResetCamera()
        fp = self.renderer.GetActiveCamera().GetFocalPoint()
        p = self.renderer.GetActiveCamera().GetPosition()
        dist = math.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1]) ** 2 + (p[2] - fp[2]) ** 2)
        self.renderer.GetActiveCamera().SetPosition(fp[2] + dist, fp[0], fp[1])
        self.renderer.GetActiveCamera().SetViewUp(0.0, 0.0, 1.0)
        self.renderer.GetActiveCamera().Zoom(1.6)
        self.render_window.Render()

    def new_images_functionality(self):
        msg = QtWidgets.QMessageBox.information(self, 'Nii images', "Select all the Nii images for training. Images MUST have an X size of 38 or 63.", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Cancel)
        if msg == QtWidgets.QMessageBox.Ok:
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            files, _ = QtWidgets.QFileDialog.getOpenFileNames(self,"Nii images", "","Nii Files (*.nii.gz)", options=options)
            if files:
                if self.validate_files(files):
                    msg2 = QtWidgets.QMessageBox.information(self, 'Mask images', "Select all the MASK Nii images for training. Images MUST have an X size of 38 or 63. Also must be selected in the same order as the nii images.", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Cancel)
                    if msg2 == QtWidgets.QMessageBox.Ok:
                        options2 = QtWidgets.QFileDialog.Options()
                        options2 |= QtWidgets.QFileDialog.DontUseNativeDialog
                        files2, _ = QtWidgets.QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","Nii Files (*.nii.gz)", options=options2)
                        if files2:
                            if self.validate_files(files2):

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
                                    # output, error = process.communicate()  # receive output from the python2 script

                                else:
                                    error3 = QtWidgets.QMessageBox.critical(self, 'Critical', "Different number of masks selected!! ", QtWidgets.QMessageBox.Ok)
                            else:
                                msg_error_size_2 = QtWidgets.QMessageBox.critical(self, 'Mask size incorrect', "Masks must have 38 or 63 size on its X axis", QtWidgets.QMessageBox.Ok)
      
                        else:
                            error2 = QtWidgets.QMessageBox.critical(self, 'Not files selected', "Not Nii files selected for new masks", QtWidgets.QMessageBox.Ok)
                else:
                    msg_error_size_1 = QtWidgets.QMessageBox.critical(self, 'Image size incorrect', "Images must have 38 or 63 size on its X axis", QtWidgets.QMessageBox.Ok)
            else:
                error1 = QtWidgets.QMessageBox.critical(self, 'Not files selected', "Not Nii files selected for new images", QtWidgets.QMessageBox.Ok)

    def new_model_functionality(self):
        self.new_network = NewNetworkWindow()
        self.new_network.show()
    
    @staticmethod
    def validate_files(files):
        for file in files:

            image_path = sitk.ReadImage(file)
            image_path_array = sitk.GetArrayFromImage(image_path)
            if image_path_array.shape[0] != 38 and image_path_array.shape[0] != 63:
                return False
        return True

    @staticmethod
    def create_new_separator():
        horizontal_line = QtWidgets.QWidget()
        horizontal_line.setFixedHeight(1)
        horizontal_line.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        horizontal_line.setStyleSheet("background-color: #c8c8c8;")
        return horizontal_line

    def process_changes(self):
        for _ in range(10):
            self.app.processEvents()
            time.sleep(0.1)


class Ui_NewNetworkWindow(object):
    def setupUi(self, NewModel):
        NewModel.setObjectName("NewModel")
        NewModel.resize(289, 347)
        self.centralwidget = QtWidgets.QWidget(NewModel)
        self.centralwidget.setObjectName("centralwidget")
        self.name_label = QtWidgets.QLabel(self.centralwidget)
        self.name_label.setGeometry(Qt.QRect(30, 40, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.name_label.setFont(font)
        self.name_label.setObjectName("name_label")
        self.z_label = QtWidgets.QLabel(self.centralwidget)
        self.z_label.setGeometry(Qt.QRect(30, 160, 21, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.onlyInt = QtGui.QIntValidator()
        self.z_label.setFont(font)
        self.z_label.setObjectName("z_label")
        self.y_label = QtWidgets.QLabel(self.centralwidget)
        self.y_label.setGeometry(Qt.QRect(30, 100, 21, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.y_label.setFont(font)
        self.y_label.setObjectName("y_label")
        self.name_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.name_edit.setGeometry(Qt.QRect(100, 40, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.name_edit.setFont(font)
        self.name_edit.setObjectName("name_edit")
        self.z_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.z_edit.setValidator(self.onlyInt)
        self.z_edit.setGeometry(Qt.QRect(100, 160, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.z_edit.setFont(font)
        self.z_edit.setObjectName("z_edit")
        self.y_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.y_edit.setValidator(self.onlyInt)
        self.y_edit.setGeometry(Qt.QRect(100, 100, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.y_edit.setFont(font)
        self.y_edit.setObjectName("y_edit")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(Qt.QRect(80, 230, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        NewModel.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(NewModel)
        self.menubar.setGeometry(Qt.QRect(0, 0, 289, 26))
        self.menubar.setObjectName("menubar")
        NewModel.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(NewModel)
        self.statusbar.setObjectName("statusbar")
        NewModel.setStatusBar(self.statusbar)

        self.retranslateUi(NewModel)
        Qt.QMetaObject.connectSlotsByName(NewModel)

    def retranslateUi(self, NewModel):
        _translate = Qt.QCoreApplication.translate
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
        # output, error = process.communicate()  # receive output from the python2 script