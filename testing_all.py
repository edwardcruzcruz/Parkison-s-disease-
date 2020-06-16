import os
import sys
import time
import numpy as np
import warnings
import scipy
import SimpleITK as sitk
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, BatchNormalization, Activation
from keras.activations import elu
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import apply_transform, transform_matrix_offset_center
K.set_image_data_format('channels_last')

rows_standard = 200
cols_standard = 200
thresh_FLAIR = 70      #to mask the brain
thresh_T1 = 30
smooth=1.


def mask_X_size(file):
    image_path = sitk.ReadImage(file)
    image_path_array = sitk.GetArrayFromImage(image_path)
    return image_path_array.shape[0]

def conv_bn_relu(nd, k=3, inputs=None):
    conv = Conv2D(nd, k, padding='same')(inputs) #, kernel_initializer='he_normal'
    #bn = BatchNormalization()(conv)
    relu = Activation('relu')(conv)
    return relu

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef_for_training(y_true, y_pred)


def dice_coef_for_training(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)


def general_preprocessing_flair(FLAIR_image):
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]
    FLAIR_image = np.float32(FLAIR_image)
    

    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    FLAIR_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)

    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
    brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
    for iii in range(np.shape(FLAIR_image)[0]):

        brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain
        #------Gaussion Normalization
    FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
    FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])

    if image_rows_Dataset >= rows_standard and image_cols_Dataset >= cols_standard:
        FLAIR_image = FLAIR_image[..., :rows_standard, :cols_standard]

    elif image_rows_Dataset >= rows_standard and image_cols_Dataset < cols_standard:
        result = np.zeros((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)

        FLAIR_image = FLAIR_image[...,:rows_standard,:]
        result[:FLAIR_image.shape[0], :FLAIR_image.shape[1], :FLAIR_image.shape[2]] = FLAIR_image
        FLAIR_image = result
    
    elif image_rows_Dataset < rows_standard and image_cols_Dataset >= cols_standard:
        result = np.zeros((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
        FLAIR_image = FLAIR_image[...,:,:cols_standard]
        result[:FLAIR_image.shape[0], :FLAIR_image.shape[1], :FLAIR_image.shape[2]] = FLAIR_image
        FLAIR_image = result

    elif image_rows_Dataset < rows_standard and image_cols_Dataset < cols_standard:
        result = np.zeros((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
        result[:FLAIR_image.shape[0], :FLAIR_image.shape[1], :FLAIR_image.shape[2]] = FLAIR_image
        FLAIR_image = result

    return FLAIR_image[..., np.newaxis], image_rows_Dataset, image_cols_Dataset

def general_postprocessing_flair(FLAIR_image, image_rows_Dataset, image_cols_Dataset):
    num_selected_slice = FLAIR_image.shape[0]
    FLAIR_image = np.squeeze(FLAIR_image, axis=3)
    if rows_standard >= image_rows_Dataset and cols_standard >= image_cols_Dataset:
        FLAIR_image = FLAIR_image[..., :image_rows_Dataset, :image_cols_Dataset]

    elif rows_standard >= image_rows_Dataset and cols_standard < image_cols_Dataset:
        result = np.zeros((num_selected_slice, image_rows_Dataset, image_cols_Dataset), dtype=np.float32)

        FLAIR_image = FLAIR_image[...,:image_rows_Dataset,:]
        result[:FLAIR_image.shape[0], :FLAIR_image.shape[1], :FLAIR_image.shape[2]] = FLAIR_image
        FLAIR_image = result
    
    elif rows_standard < image_rows_Dataset and cols_standard >= image_cols_Dataset:
        result = np.zeros((num_selected_slice, image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
        FLAIR_image = FLAIR_image[...,:,:image_cols_Dataset]
        result[:FLAIR_image.shape[0], :FLAIR_image.shape[1], :FLAIR_image.shape[2]] = FLAIR_image
        FLAIR_image = result

    elif image_rows_Dataset < rows_standard and image_cols_Dataset < cols_standard:
        result = np.zeros((num_selected_slice, image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
        result[:FLAIR_image.shape[0], :FLAIR_image.shape[1], :FLAIR_image.shape[2]] = FLAIR_image
        FLAIR_image = result

    return FLAIR_image

def generateMaskOneArgument(FLAIR_image, model_path, resultPath):

    FLAIR_image = sitk.ReadImage(FLAIR_image)
    FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)

    imgs_test, Y_original, Z_original = general_preprocessing_flair(FLAIR_array)
    flair_image = imgs_test[..., 0:1].copy()
    imgs_test = flair_image
    img_shape = (rows_standard, cols_standard, 1)
    model = get_unet_2(img_shape, first5=True)
    model.load_weights(model_path)
    pred = model.predict(imgs_test, batch_size=1, verbose=False)
    pred[pred > 0.5] = 1.
    pred[pred <= 0.5] = 0.
    original_pred = general_postprocessing_flair(pred, Y_original, Z_original)

    filename_resultImage = resultPath
    sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage )

def get_unet_2(img_shape = None, first5=True):
    inputs = Input(shape = img_shape)
    concat_axis = -1

    if first5: filters = 5
    else: filters = 3
    conv1 = conv_bn_relu(64, filters, inputs)
    conv1 = conv_bn_relu(64, filters, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv3 = conv_bn_relu(128, 3, pool1)
    conv3 = conv_bn_relu(128, 3, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_bn_relu(256, 3, pool3)
    conv4 = conv_bn_relu(256, 4, conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_bn_relu(512, 3, pool4)
    conv5 = conv_bn_relu(512, 3, conv5)

    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = conv_bn_relu(256, 3, up6)
    conv6 = conv_bn_relu(256, 3, conv6)

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = conv_bn_relu(128, 3, up7)
    conv7 = conv_bn_relu(128, 3, conv7)

    up_conv8 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = conv_bn_relu(64, 3, up9)
    conv9 = conv_bn_relu(64, 3, conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=(ch, cw))(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9) #, kernel_initializer='he_normal'
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)

    return model

def main():
    if not os.path.exists('output/'):
        os.makedirs('output')
    if not os.path.exists('input/'):
        os.makedirs('input/pre')
        os.makedirs('input/orig')

    model_path = 'models/unet2_onlyFlair/_complete__onlyFlair_unet2.h5'
    input_dir = '../../Data/flair_pre/'


    for file in os.listdir(input_dir):
        original_path = os.path.abspath(os.path.join(input_dir, file))
        print(original_path)
        fileName = file.split(".")[0]
        outpath_path = 'output/' + fileName + "_output.nii.gz"
        generateMaskOneArgument(original_path, model_path, outpath_path )

if __name__ == "__main__":
    main()


