import os
import numpy as np
import SimpleITK as sitk
import warnings

def converting_masks(array_of_masks, patient):
    if patient < 20:
        name = "_Utrecht_"
        mask= array_of_masks[patient*38:(patient+1)*38, ...]
    elif patient < 40:
        name = "_Singapore_"
        mask = array_of_masks[patient*38:(patient+1)*38, ...]
    else:
        name = "_GE3T_"
        mask = array_of_masks[1520+(patient-40)*63:1520+(patient-39)*63, ...]
    print mask.shape
    model_path = 'results/'
    filename_resultImage = model_path + str(patient) + name + '_5.nii.gz'
    sitk.WriteImage(sitk.GetImageFromArray(mask), filename_resultImage )

def converting_images(array_of_images, patient):
    if patient < 20:
        name = "_Utrecht_"
        imageTempFlair = array_of_images[patient*38:(patient+1)*38, :, :, 0]
        imageFlair = imageTempFlair[..., np.newaxis]

        imageTempT1 = array_of_images[patient*38:(patient+1)*38, :, :, 1]
        imageT1 = imageTempT1[..., np.newaxis]
    elif patient < 40:
        name = "_Singapore_"
        imageTempFlair = array_of_images[patient*38:(patient+1)*38, :, :, 0]
        imageFlair = imageTempFlair[..., np.newaxis]

        imageTempT1 = array_of_images[patient*38:(patient+1)*38, :, :, 1]
        imageT1 = imageTempT1[..., np.newaxis]
    else:
        name = "_GE3T_"
        imageTempFlair = array_of_images[1520+(patient-40)*63:1520+(patient-39)*63, :, :, 0]
        imageFlair = imageTempFlair[..., np.newaxis]

        imageTempT1 = array_of_images[1520+(patient-40)*63:1520+(patient-39)*63, :, :, 1]
        imageT1 = imageTempT1[..., np.newaxis]

    print imageT1.shape
    print imageFlair.shape

    model_path_flair = 'flair_pre/'
    filename_resultImage_flair = model_path_flair + str(patient) + name + '_4.nii.gz'
    sitk.WriteImage(sitk.GetImageFromArray(imageFlair), filename_resultImage_flair )

    model_path_t1 = 't1_pre/'
    filename_resultImage_t1 = model_path_t1 + str(patient) + name + '_3.nii.gz'
    sitk.WriteImage(sitk.GetImageFromArray(imageT1), filename_resultImage_t1 )

#leave-one-out evaluation
def main():
    warnings.filterwarnings("ignore")
    patient_num  = 60
    masks = np.load('data/masks_three_datasets_sorted.npy')
    images = np.load('data/images_three_datasets_sorted.npy')
    for patient in range(0, patient_num):
        # converting_images(masks, patient)
        converting_images(images, patient)

if __name__=='__main__':
    main()
