import os
import SimpleITK as sitk
import numpy as np
import sys

def multiply_images(flair_image, t1_image, patient, type='raw'):
    FLAIR_array = sitk.GetArrayFromImage(flair_image).astype(float)
    T1_array = sitk.GetArrayFromImage(t1_image).astype(float)

    multiplied = np.multiply(FLAIR_array, T1_array)
    if patient < 20:
        name = "_Utrecht_"
    elif patient < 40:
        name = "_Singapore_"
    else:
        name = "_GE3T_"
    path = 'multiplication/' + type
    typeFile = "_7" if type == "raw" else "_9"
    filename_resultImage = path + name + str(patient) + typeFile + '.nii.gz'
    sitk.WriteImage(sitk.GetImageFromArray(multiplied), filename_resultImage )

def dot_product_images(flair_image, t1_image, patient, type='raw'):
    FLAIR_array = sitk.GetArrayFromImage(flair_image)
    T1_array = sitk.GetArrayFromImage(t1_image)

    dotProduct = np.dot(FLAIR_array, T1_array)
    if patient < 20:
        name = "_Utrecht_"
    elif patient < 40:
        name = "_Singapore_"
    else:
        name = "_GE3T_"
    path = 'dots/'  + type
    typeFile = "_6" if type == "raw" else "_8"
    filename_resultImage = path + name + str(patient) + typeFile + '.nii.gz'
    sitk.WriteImage(sitk.GetImageFromArray(dotProduct), filename_resultImage )


def main(): #for normal data
    patient_num = 60
    for patient in range(0,patient_num):
        if patient < 20: dir = '../../Data/originalData/Utrecht/'
        elif patient < 40: dir = '../../Data/originalData/Singapore/'
        else: dir = '../../Data/originalData/GE3T/'
        dirs = os.listdir(dir)
        dirs.sort()
        dir += dirs[patient%20]
        FLAIR_image = sitk.ReadImage(dir + '/pre/FLAIR.nii.gz')
        T1_image = sitk.ReadImage(dir + '/pre/T1.nii.gz')
        # multiply_images(FLAIR_image, T1_image, patient)
        dot_product_images(FLAIR_image, T1_image, patient)

def main2(): #for preprocessed data
    patient_num = 60
    for patient in range(0,patient_num):
        theFileFlair = 'flair_pre/'
        theFileT1 = 't1_pre/'
        if patient < 20: name = '_Utrecht_.nii.gz'
        elif patient < 40: name = '_Singapore_.nii.gz'
        else: name = '_GE3T_.nii.gz'

        FLAIR_image = sitk.ReadImage(theFileFlair  + str(patient) + name)
        T1_image = sitk.ReadImage(theFileT1 + str(patient) + name)
        multiply_images(FLAIR_image, T1_image, patient, type='pre')
        dot_product_images(FLAIR_image, T1_image, patient, type='pre')


if __name__=='__main__':
    args = sys.argv
    print args
    election = int(args[1])
    if election == 1:
        main()
    else:
        main2()
    #main()