#Codes for validating the WMH Segmetation Challenge public training Datasets. The algorithm won the MICCAI WMH Segmentation Challenge 2017.
#Codes are written by Mr. Hongwei Li (h.l.li@dundee.ac.uk; hongwei.li@tum.de) and Mr. Gongfa Jiang (jianggfa@mail2.sysu.edu.cn). They are PhD students in Technical University of Munich and Sun Yat-sen University.
#Please cite our paper titled 'Fully Convolutional Networks Ensembles for White Matter Hyperintensities Segmentation in MR Images' if you found it is useful to your research.
#Please contact me if there is any bug you want to report or any details you would like to know. 

import os
import time
import scipy
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, BatchNormalization, Activation
from keras.activations import elu
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import apply_transform, transform_matrix_offset_center
import SimpleITK as sitk
import warnings
K.set_image_data_format('channels_last')

smooth=1.
rows_standard = 200
cols_standard = 200
thresh_FLAIR = 70      #to mask the brain
thresh_T1 = 30


def dice_coef_for_training(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef_for_training(y_true, y_pred)

def conv_bn_relu(nd, k=3, inputs=None):
    conv = Conv2D(nd, k, padding='same')(inputs) #, kernel_initializer='he_normal'
    #bn = BatchNormalization()(conv)
    relu = Activation('relu')(conv)
    return relu

def conv_elu(nd, k=3, inputs=None):
    conv = Conv2D(nd, k, padding='same', kernel_initializer='he_normal')(inputs) #, kernel_initializer='he_normal'
    eluVar = Activation('elu')(conv)
    return eluVar

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
#define U-Net architecture
def get_unet(img_shape = None, first5=True):
    inputs = Input(shape = img_shape)
    concat_axis = -1

    if first5: filters = 5
    else: filters = 3
    conv1 = conv_bn_relu(64, filters, inputs)
    conv1 = conv_bn_relu(64, filters, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_bn_relu(96, 3, pool1)
    conv2 = conv_bn_relu(96, 3, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_bn_relu(128, 3, pool2)
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

    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = conv_bn_relu(96, 3, up8)
    conv8 = conv_bn_relu(96, 3, conv8)

    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
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

def get_unet_3(img_shape = None, first5=True):
    inputs = Input(shape = img_shape)
    concat_axis = -1

    if first5: filters = 5
    else: filters = 3
    conv1 = conv_elu(64, filters, inputs)
    conv1 = conv_elu(64, filters, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv3 = conv_elu(128, 3, pool1)
    conv3 = conv_elu(128, 3, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_elu(256, 3, pool3)
    conv4 = conv_elu(256, 4, conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_elu(512, 3, pool4)
    conv5 = conv_elu(512, 3, conv5)

    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = conv_elu(256, 3, up6)
    conv6 = conv_elu(256, 3, conv6)

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = conv_elu(128, 3, up7)
    conv7 = conv_elu(128, 3, conv7)

    up_conv8 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = conv_elu(64, 3, up9)
    conv9 = conv_elu(64, 3, conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=(ch, cw))(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv9) #, kernel_initializer='he_normal'
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)

    return model

def get_unet_3_softmax(img_shape = None, first5=True):
    inputs = Input(shape = img_shape)
    concat_axis = -1

    if first5: filters = 5
    else: filters = 3
    conv1 = conv_elu(64, filters, inputs)
    conv1 = conv_elu(64, filters, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv3 = conv_elu(128, 3, pool1)
    conv3 = conv_elu(128, 3, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_elu(256, 3, pool3)
    conv4 = conv_elu(256, 4, conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_elu(512, 3, pool4)
    conv5 = conv_elu(512, 3, conv5)

    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = conv_elu(256, 3, up6)
    conv6 = conv_elu(256, 3, conv6)

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = conv_elu(128, 3, up7)
    conv7 = conv_elu(128, 3, conv7)

    up_conv8 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = conv_elu(64, 3, up9)
    conv9 = conv_elu(64, 3, conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=(ch, cw))(conv9)
    conv10 = Conv2D(1, 1, activation='softmax', padding='same', kernel_initializer='he_normal')(conv9) #, kernel_initializer='he_normal'
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)

    return model

def augmentation(x_0, x_1, y):
    theta = (np.random.uniform(-15, 15) * np.pi) // 180.
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    shear = np.random.uniform(-.1, .1)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                                [0, np.cos(shear), 0],
                                [0, 0, 1]])
    zx, zy = np.random.uniform(.9, 1.1, 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    augmentation_matrix = np.dot(np.dot(rotation_matrix, shear_matrix), zoom_matrix)
    transform_matrix = transform_matrix_offset_center(augmentation_matrix, x_0.shape[0], x_0.shape[1])
    x_0 = apply_transform(x_0[..., np.newaxis], transform_matrix, channel_axis=2)
    x_1 = apply_transform(x_1[..., np.newaxis], transform_matrix, channel_axis=2)
    y = apply_transform(y[..., np.newaxis], transform_matrix, channel_axis=2)
    return x_0[..., 0], x_1[..., 0], y[..., 0]

def augmentation_2(x_0, y):
    theta = (np.random.uniform(-15, 15) * np.pi) // 180.
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    shear = np.random.uniform(-.1, .1)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                                [0, np.cos(shear), 0],
                                [0, 0, 1]])
    zx, zy = np.random.uniform(.9, 1.1, 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    augmentation_matrix = np.dot(np.dot(rotation_matrix, shear_matrix), zoom_matrix)
    transform_matrix = transform_matrix_offset_center(augmentation_matrix, x_0.shape[0], x_0.shape[1])
    x_0 = apply_transform(x_0[..., np.newaxis], transform_matrix, channel_axis=2)
    y = apply_transform(y[..., np.newaxis], transform_matrix, channel_axis=2)
    return x_0[..., 0], y[..., 0]

def Utrecht_preprocessing(FLAIR_image, T1_image):

    channel_num = 2
    #print(np.shape(FLAIR_image))
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]
    T1_image = np.float32(T1_image)

    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
    imgs_mask_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard,1), dtype=np.float32)

    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
    brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
    for iii in range(np.shape(FLAIR_image)[0]):
        brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain
    FLAIR_image = FLAIR_image[:, (image_rows_Dataset/2-rows_standard/2):(image_rows_Dataset/2+rows_standard/2), (image_cols_Dataset/2-cols_standard/2):(image_cols_Dataset/2+cols_standard/2)]
    brain_mask_FLAIR = brain_mask_FLAIR[:, (image_rows_Dataset/2-rows_standard/2):(image_rows_Dataset/2+rows_standard/2), (image_cols_Dataset/2-cols_standard/2):(image_cols_Dataset/2+cols_standard/2)]
    ###------Gaussion Normalization here
    FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
    FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])
    # T1 -----------------------------------------------
    brain_mask_T1[T1_image >=thresh_T1] = 1
    brain_mask_T1[T1_image < thresh_T1] = 0
    for iii in range(np.shape(T1_image)[0]):
        brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
    T1_image = T1_image[:, (image_rows_Dataset/2-rows_standard/2):(image_rows_Dataset/2+rows_standard/2), (image_cols_Dataset/2-cols_standard/2):(image_cols_Dataset/2+cols_standard/2)]
    brain_mask_T1 = brain_mask_T1[:, (image_rows_Dataset/2-rows_standard/2):(image_rows_Dataset/2+rows_standard/2), (image_cols_Dataset/2-cols_standard/2):(image_cols_Dataset/2+cols_standard/2)]
    #------Gaussion Normalization
    T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      
    T1_image /=np.std(T1_image[brain_mask_T1 == 1])
    #---------------------------------------------------
    FLAIR_image  = FLAIR_image[..., np.newaxis]
    T1_image  = T1_image[..., np.newaxis]
    imgs_two_channels = np.concatenate((FLAIR_image, T1_image), axis = 3)
    #print(np.shape(imgs_two_channels))
    return imgs_two_channels

def GE3T_preprocessing(FLAIR_image, T1_image):
    channel_num = 2
    start_cut = 46
    num_selected_slice = np.shape(FLAIR_image)[0]
    print num_selected_slice
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]
    FLAIR_image = np.float32(FLAIR_image)
    T1_image = np.float32(T1_image)

    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
    imgs_mask_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard,1), dtype=np.float32)
    FLAIR_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
    T1_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)

    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
    brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
    for iii in range(np.shape(FLAIR_image)[0]):
  
        brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain
        #------Gaussion Normalization
    FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
    FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])

    FLAIR_image_suitable[...] = np.min(FLAIR_image)
    FLAIR_image_suitable[:, :, (cols_standard/2-image_cols_Dataset/2):(cols_standard/2+image_cols_Dataset/2)] = FLAIR_image[:, start_cut:start_cut+rows_standard, :]
   
    # T1 -----------------------------------------------
    brain_mask_T1[T1_image >=thresh_T1] = 1
    brain_mask_T1[T1_image < thresh_T1] = 0
    for iii in range(np.shape(T1_image)[0]):
 
        brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
        #------Gaussion Normalization
    T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      #Gaussion Normalization
    T1_image /=np.std(T1_image[brain_mask_T1 == 1])

    T1_image_suitable[...] = np.min(T1_image)
    T1_image_suitable[:, :, (cols_standard-image_cols_Dataset)/2:(cols_standard+image_cols_Dataset)/2] = T1_image[:, start_cut:start_cut+rows_standard, :]
    #---------------------------------------------------
    FLAIR_image_suitable  = FLAIR_image_suitable[..., np.newaxis]
    T1_image_suitable  = T1_image_suitable[..., np.newaxis]
    
    imgs_two_channels = np.concatenate((FLAIR_image_suitable, T1_image_suitable), axis = 3)
    #print(np.shape(imgs_two_channels))
    return imgs_two_channels


#train single model on the training set
# flair: Use Flair data
# T1: Use T1 data
# patient: the test patient, the one to be deleted
# full: use all data or per scanner data
# first5: use 5 filters or 3 filters at beggining of the net
# aug: use data augmentation
# verbose: use verbose in net

def train_leave_one_out(images, masks, patient=0, flair=True, t1=True, full=True, first5=True, aug=True, verbose=False):
    if full:
        if patient < 40:
            images = np.delete(images, range(patient*38, (patient+1)*38), axis=0)
            masks = np.delete(masks, range(patient*38, (patient+1)*38), axis=0)
        else:
            images = np.delete(images, range(1520+(patient-40)*63, 1520+(patient-39)*63), axis=0)
            masks = np.delete(masks, range(1520+(patient-40)*63, 1520+(patient-39)*63), axis=0)
    else:
        if patient < 20:
            images = images[:760, ...]
            masks = masks[:760, ...]
            images = np.delete(images, range(patient*38, (patient+1)*38), axis=0)
            masks = np.delete(masks, range(patient*38, (patient+1)*38), axis=0)
        elif patient < 40:
            images = images[760:1520, ...]
            masks = masks[760:1520, ...]
            images = np.delete(images, range((patient-20)*38, (patient-19)*38), axis=0)
            masks = np.delete(masks, range((patient-20)*38, (patient-19)*38), axis=0)
        else:
            images = images[1520:, ...]
            masks = masks[1520:, ...]
            images = np.delete(images, range((patient-40)*63, (patient-39)*63), axis=0)
            masks = np.delete(masks, range((patient-40)*63, (patient-39)*63), axis=0)
    if aug:
        images = np.concatenate((images, images[..., ::-1, :]), axis=0)
        masks = np.concatenate((masks, masks[..., ::-1, :]), axis=0)
    samples_num = images.shape[0]
    print(samples_num)
    row = images.shape[1]
    col = images.shape[2]
    if aug: epoch = 50
    else: epoch = 200
    batch_size = 30

    img_shape = (row, col, flair+t1)
    model = get_unet(img_shape, first5)
    current_epoch = 1
    while current_epoch <= epoch:
        print('Epoch ', str(current_epoch), '/', str(epoch))
        if aug:
            images_aug = np.zeros(images.shape, dtype=np.float32)
            masks_aug = np.zeros(masks.shape, dtype=np.float32)
            for i in range(samples_num):
                images_aug[i, ..., 0], images_aug[i, ..., 1], masks_aug[i, ..., 0] = augmentation(images[i, ..., 0], images[i, ..., 1], masks[i, ..., 0])
            image = np.concatenate((images, images_aug), axis=0)
            mask = np.concatenate((masks, masks_aug), axis=0)
        else:
            image = images.copy()
            mask = masks.copy()
        if not flair: image = image[..., 1:2].copy()
        if not t1: image = image[..., 0:1].copy()
        history = model.fit(image, mask, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        current_epoch += 1
        if history.history['loss'][-1] > 0.99:
            model = get_unet(img_shape, first5)
            current_epoch = 1
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if full: model_path += 'Full_'
    else:
        if patient < 20: model_path += 'Utrecht_'
        elif patient < 40: model_path += 'Singapore_'
        else: model_path += 'GE3T_'
    if flair: model_path += 'Flair_'
    if t1: model_path += 'T1_'
    if first5: model_path += '5_'
    else: model_path += '3_'
    if aug: model_path += 'Augmentation/'
    else: model_path += 'No_Augmentation/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path += str(patient) + '.h5'
    model.save_weights(model_path)
    print('Model saved to ', model_path)

def train_all_85_15(images, masks, flair=True, t1=True, first5=True, aug=True, verbose=False):
    images_delete = [5,7,19,25,26,32] # actual images will be 5, 8, 21, 28, 30, 37
    images_delete_2 = [41,50,51] # 47, 57, 58
    for test_1 in images_delete:
        images = np.delete(images, range(test_1*38, (test_1+1)*38), axis=0)
        masks = np.delete(masks, range(test_1*38, (test_1+1)*38), axis=0)
    for test_2 in images_delete_2:
        images = np.delete(images, range(1520+(test_2-40)*63, 1520+(test_2-39)*63), axis=0)
        masks = np.delete(masks, range(1520+(test_2-40)*63, 1520+(test_2-39)*63), axis=0)
    if aug:
        images = np.concatenate((images, images[..., ::-1, :]), axis=0)
        masks = np.concatenate((masks, masks[..., ::-1, :]), axis=0)
    samples_num = images.shape[0]
    print(samples_num)
    row = images.shape[1]
    col = images.shape[2]
    if aug: epoch = 50
    else: epoch = 200
    batch_size = 30

    img_shape = (row, col, flair+t1)
    model = get_unet(img_shape, first5)
    current_epoch = 1
    while current_epoch <= epoch:
        print('Epoch ', str(current_epoch), '/', str(epoch))
        if aug:
            images_aug = np.zeros(images.shape, dtype=np.float32)
            masks_aug = np.zeros(masks.shape, dtype=np.float32)
            for i in range(samples_num):
                images_aug[i, ..., 0], images_aug[i, ..., 1], masks_aug[i, ..., 0] = augmentation(images[i, ..., 0], images[i, ..., 1], masks[i, ..., 0])
            image = np.concatenate((images, images_aug), axis=0)
            mask = np.concatenate((masks, masks_aug), axis=0)
        else:
            image = images.copy()
            mask = masks.copy()
        if not flair: image = image[..., 1:2].copy()
        if not t1: image = image[..., 0:1].copy()
        history = model.fit(image, mask, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        current_epoch += 1
        if history.history['loss'][-1] > 0.99:
            model = get_unet(img_shape, first5)
            current_epoch = 1
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path += 'Full_'
    model_path += ''
    if flair: model_path += 'Flair_'
    if t1: model_path += 'T1_'
    if first5: model_path += '5_'
    else: model_path += '3_'
    if aug: model_path += 'Augmentation/'
    else: model_path += 'No_Augmentation/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path += str('_85_15') + '.h5'
    model.save_weights(model_path)
    print('Model saved to ', model_path)

def train_only_mult_data(images, masks, patient=0, full=True, first5=True, aug=True, verbose=False):
    if full:
        if patient < 40:
            print 'entering here'
            print images.shape
            print masks.shape
            images = np.delete(images, range(patient*38, (patient+1)*38), axis=0)
            masks = np.delete(masks, range(patient*38, (patient+1)*38), axis=0)
        else:
            images = np.delete(images, range(1520+(patient-40)*63, 1520+(patient-39)*63), axis=0)
            masks = np.delete(masks, range(1520+(patient-40)*63, 1520+(patient-39)*63), axis=0)
    else:
        if patient < 20:
            images = images[:760, ...]
            masks = masks[:760, ...]
            images = np.delete(images, range(patient*38, (patient+1)*38), axis=0)
            masks = np.delete(masks, range(patient*38, (patient+1)*38), axis=0)
        elif patient < 40:
            images = images[760:1520, ...]
            masks = masks[760:1520, ...]
            images = np.delete(images, range((patient-20)*38, (patient-19)*38), axis=0)
            masks = np.delete(masks, range((patient-20)*38, (patient-19)*38), axis=0)
        else:
            images = images[1520:, ...]
            masks = masks[1520:, ...]
            images = np.delete(images, range((patient-40)*63, (patient-39)*63), axis=0)
            masks = np.delete(masks, range((patient-40)*63, (patient-39)*63), axis=0)
    if aug:
        images = np.concatenate((images, images[..., ::-1, :]), axis=0)
        masks = np.concatenate((masks, masks[..., ::-1, :]), axis=0)
    samples_num = images.shape[0]
    print(samples_num)
    row = images.shape[1]
    col = images.shape[2]
    if aug: epoch = 50
    else: epoch = 200
    batch_size = 30

    print images.shape
    print masks.shape
    img_shape = (row, col, 1)
    model = get_unet(img_shape, first5)
    current_epoch = 1
    while current_epoch <= epoch:
        print('Epoch ', str(current_epoch), '/', str(epoch))
        if aug:
            images_aug = np.zeros(images.shape, dtype=np.float32)
            masks_aug = np.zeros(masks.shape, dtype=np.float32)
            for i in range(samples_num):
                images_aug[i, ..., 0], masks_aug[i, ..., 0] = augmentation_2(images[i, ..., 0], masks[i, ..., 0])
            image = np.concatenate((images, images_aug), axis=0)
            mask = np.concatenate((masks, masks_aug), axis=0)
        else:
            image = images.copy()
            mask = masks.copy()
        image = image[..., 0:1].copy()
        print image.shape
        print mask.shape
        history = model.fit(image, mask, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        current_epoch += 1
        if history.history['loss'][-1] > 0.99:
            model = get_unet(img_shape, first5)
            current_epoch = 1
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if full: model_path += 'Full_'
    else:
        if patient < 20: model_path += 'Utrecht_'
        elif patient < 40: model_path += 'Singapore_'
        else: model_path += 'GE3T_'
    if first5: model_path += '5_'
    else: model_path += '3_'
    if aug: model_path += 'Augmentation/'
    else: model_path += 'No_Augmentation/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path += str(patient)
    model_path += '_multData' + '.h5'
    model.save_weights(model_path)
    print('Model saved to ', model_path)

def train_only_mult_data_85_15(images, masks, flair=True, t1=True, first5=True, aug=True, verbose=False):
    images_delete = [5,7,19,25,26,32] # actual images will be 5, 8, 21, 28, 30, 37
    images_delete_2 = [41,50,51] # 47, 57, 58
    for test_1 in images_delete:
        images = np.delete(images, range(test_1*38, (test_1+1)*38), axis=0)
        masks = np.delete(masks, range(test_1*38, (test_1+1)*38), axis=0)
    for test_2 in images_delete_2:
        images = np.delete(images, range(1520+(test_2-40)*63, 1520+(test_2-39)*63), axis=0)
        masks = np.delete(masks, range(1520+(test_2-40)*63, 1520+(test_2-39)*63), axis=0)
    if aug:
        images = np.concatenate((images, images[..., ::-1, :]), axis=0)
        masks = np.concatenate((masks, masks[..., ::-1, :]), axis=0)
    samples_num = images.shape[0]
    print(samples_num)
    row = images.shape[1]
    col = images.shape[2]
    if aug: epoch = 50
    else: epoch = 200
    batch_size = 30

    print images.shape
    print masks.shape
    img_shape = (row, col, 1)
    model = get_unet(img_shape, first5)
    current_epoch = 1
    while current_epoch <= epoch:
        print('Epoch ', str(current_epoch), '/', str(epoch))
        if aug:
            images_aug = np.zeros(images.shape, dtype=np.float32)
            masks_aug = np.zeros(masks.shape, dtype=np.float32)
            for i in range(samples_num):
                images_aug[i, ..., 0], masks_aug[i, ..., 0] = augmentation_2(images[i, ..., 0], masks[i, ..., 0])
            image = np.concatenate((images, images_aug), axis=0)
            mask = np.concatenate((masks, masks_aug), axis=0)
        else:
            image = images.copy()
            mask = masks.copy()
        image = image[..., 0:1].copy()
        print image.shape
        print mask.shape
        history = model.fit(image, mask, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        current_epoch += 1
        if history.history['loss'][-1] > 0.99:
            model = get_unet(img_shape, first5)
            current_epoch = 1
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if first5: model_path += '5_'
    else: model_path += '3_'
    if aug: model_path += 'Augmentation/'
    else: model_path += 'No_Augmentation/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path += '_complete_'
    model_path += '_multData' + '.h5'
    model.save_weights(model_path)
    print('Model saved to ', model_path)

def train_only_85_15_flair(images, masks, first5=True, aug=True, verbose=False):
    images_delete = [5,7,19,25,26,32] # actual images will be 5, 8, 21, 28, 30, 37
    images_delete_2 = [41,50,51] # 47, 57, 58
    for test_1 in images_delete:
        images = np.delete(images, range(test_1*38, (test_1+1)*38), axis=0)
        masks = np.delete(masks, range(test_1*38, (test_1+1)*38), axis=0)
    for test_2 in images_delete_2:
        images = np.delete(images, range(1520+(test_2-40)*63, 1520+(test_2-39)*63), axis=0)
        masks = np.delete(masks, range(1520+(test_2-40)*63, 1520+(test_2-39)*63), axis=0)
    if aug:
        images = np.concatenate((images, images[..., ::-1, :]), axis=0)
        masks = np.concatenate((masks, masks[..., ::-1, :]), axis=0)
    samples_num = images.shape[0]
    print(samples_num)
    row = images.shape[1]
    col = images.shape[2]
    if aug: epoch = 50
    else: epoch = 200
    batch_size = 30

    print images.shape
    print masks.shape
    img_shape = (row, col, 1)
    model = get_unet(img_shape, first5)
    current_epoch = 1
    while current_epoch <= epoch:
        print('Epoch ', str(current_epoch), '/', str(epoch))
        if aug:
            images_aug = np.zeros(images.shape, dtype=np.float32)
            masks_aug = np.zeros(masks.shape, dtype=np.float32)
            for i in range(samples_num):
                images_aug[i, ..., 0], masks_aug[i, ..., 0] = augmentation_2(images[i, ..., 0], masks[i, ..., 0])
            image = np.concatenate((images, images_aug), axis=0)
            mask = np.concatenate((masks, masks_aug), axis=0)
        else:
            image = images.copy()
            mask = masks.copy()
        image = image[..., 0:1].copy()
        print image.shape
        print mask.shape
        history = model.fit(image, mask, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        current_epoch += 1
        if history.history['loss'][-1] > 0.99:
            model = get_unet(img_shape, first5)
            current_epoch = 1
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if first5: model_path += '5_'
    else: model_path += '3_'
    if aug: model_path += 'Augmentation/'
    else: model_path += 'No_Augmentation/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path += '_complete_'
    model_path += '_onlyFlair' + '.h5'
    model.save_weights(model_path)
    print('Model saved to ', model_path)

def train_all_85_15_unet_2(images, masks, flair=True, t1=True, first5=True, aug=True, verbose=False):
    images_delete = [5,7,19,25,26,32] # actual images will be 5, 8, 21, 28, 30, 37
    images_delete_2 = [41,50,51] # 47, 57, 58
    for test_1 in images_delete:
        images = np.delete(images, range(test_1*38, (test_1+1)*38), axis=0)
        masks = np.delete(masks, range(test_1*38, (test_1+1)*38), axis=0)
    for test_2 in images_delete_2:
        images = np.delete(images, range(1520+(test_2-40)*63, 1520+(test_2-39)*63), axis=0)
        masks = np.delete(masks, range(1520+(test_2-40)*63, 1520+(test_2-39)*63), axis=0)
    if aug:
        images = np.concatenate((images, images[..., ::-1, :]), axis=0)
        masks = np.concatenate((masks, masks[..., ::-1, :]), axis=0)
    samples_num = images.shape[0]
    print(samples_num)
    row = images.shape[1]
    col = images.shape[2]
    if aug: epoch = 50
    else: epoch = 200
    batch_size = 30

    img_shape = (row, col, flair+t1)
    model = get_unet_2(img_shape, first5)
    current_epoch = 1
    while current_epoch <= epoch:
        print('Epoch ', str(current_epoch), '/', str(epoch))
        if aug:
            images_aug = np.zeros(images.shape, dtype=np.float32)
            masks_aug = np.zeros(masks.shape, dtype=np.float32)
            for i in range(samples_num):
                images_aug[i, ..., 0], images_aug[i, ..., 1], masks_aug[i, ..., 0] = augmentation(images[i, ..., 0], images[i, ..., 1], masks[i, ..., 0])
            image = np.concatenate((images, images_aug), axis=0)
            mask = np.concatenate((masks, masks_aug), axis=0)
        else:
            image = images.copy()
            mask = masks.copy()
        if not flair: image = image[..., 1:2].copy()
        if not t1: image = image[..., 0:1].copy()
        history = model.fit(image, mask, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        current_epoch += 1
        if history.history['loss'][-1] > 0.99:
            model = get_unet_2(img_shape, first5)
            current_epoch = 1
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path += 'Full_'
    model_path += ''
    if flair: model_path += 'Flair_'
    if t1: model_path += 'T1_'
    if first5: model_path += '5_'
    else: model_path += '3_'
    if aug: model_path += 'Augmentation/'
    else: model_path += 'No_Augmentation/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path += str('_85_15_unet2') + '.h5'
    model.save_weights(model_path)
    print('Model saved to ', model_path)

def train_all_85_15_unet_2_flair(images, masks, flair=True, t1=True, first5=True, aug=True, verbose=False):
    images_delete = [5,7,19,25,26,32] # actual images will be 5, 8, 21, 28, 30, 37
    images_delete_2 = [41,50,51] # 47, 57, 58
    for test_1 in images_delete:
        images = np.delete(images, range(test_1*38, (test_1+1)*38), axis=0)
        masks = np.delete(masks, range(test_1*38, (test_1+1)*38), axis=0)
    for test_2 in images_delete_2:
        images = np.delete(images, range(1520+(test_2-40)*63, 1520+(test_2-39)*63), axis=0)
        masks = np.delete(masks, range(1520+(test_2-40)*63, 1520+(test_2-39)*63), axis=0)
    if aug:
        images = np.concatenate((images, images[..., ::-1, :]), axis=0)
        masks = np.concatenate((masks, masks[..., ::-1, :]), axis=0)
    samples_num = images.shape[0]
    print(samples_num)
    row = images.shape[1]
    col = images.shape[2]
    if aug: epoch = 50
    else: epoch = 200
    batch_size = 30

    img_shape = (row, col, 1)
    model = get_unet_2(img_shape, first5)
    current_epoch = 1
    while current_epoch <= epoch:
        print('Epoch ', str(current_epoch), '/', str(epoch))
        if aug:
            images_aug = np.zeros(images.shape, dtype=np.float32)
            masks_aug = np.zeros(masks.shape, dtype=np.float32)
            for i in range(samples_num):
                images_aug[i, ..., 0], masks_aug[i, ..., 0] = augmentation_2(images[i, ..., 0], masks[i, ..., 0])
            image = np.concatenate((images, images_aug), axis=0)
            mask = np.concatenate((masks, masks_aug), axis=0)
        else:
            image = images.copy()
            mask = masks.copy()
        image = image[..., 0:1].copy()
        history = model.fit(image, mask, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        current_epoch += 1
        if history.history['loss'][-1] > 0.99:
            model = get_unet_2(img_shape, first5)
            current_epoch = 1
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if first5: model_path += '5_'
    else: model_path += '3_'
    if aug: model_path += 'Augmentation/'
    else: model_path += 'No_Augmentation/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path += '_complete_'
    model_path += '_onlyFlair_unet2' + '.h5'
    model.save_weights(model_path)
    print('Model saved to ', model_path)

def train_all_85_15_unet_3(images, masks, flair=True, t1=True, first5=True, aug=True, verbose=False):
    images_delete = [5,7,19,25,26,32] # actual images will be 5, 8, 21, 28, 30, 37
    images_delete_2 = [41,50,51] # 47, 57, 58
    for test_1 in images_delete:
        images = np.delete(images, range(test_1*38, (test_1+1)*38), axis=0)
        masks = np.delete(masks, range(test_1*38, (test_1+1)*38), axis=0)
    for test_2 in images_delete_2:
        images = np.delete(images, range(1520+(test_2-40)*63, 1520+(test_2-39)*63), axis=0)
        masks = np.delete(masks, range(1520+(test_2-40)*63, 1520+(test_2-39)*63), axis=0)
    if aug:
        images = np.concatenate((images, images[..., ::-1, :]), axis=0)
        masks = np.concatenate((masks, masks[..., ::-1, :]), axis=0)
    samples_num = images.shape[0]
    print(samples_num)
    row = images.shape[1]
    col = images.shape[2]
    if aug: epoch = 50
    else: epoch = 200
    batch_size = 30

    img_shape = (row, col, flair+t1)
    model = get_unet_3(img_shape, first5)
    current_epoch = 1
    while current_epoch <= epoch:
        print('Epoch ', str(current_epoch), '/', str(epoch))
        if aug:
            images_aug = np.zeros(images.shape, dtype=np.float32)
            masks_aug = np.zeros(masks.shape, dtype=np.float32)
            for i in range(samples_num):
                images_aug[i, ..., 0], images_aug[i, ..., 1], masks_aug[i, ..., 0] = augmentation(images[i, ..., 0], images[i, ..., 1], masks[i, ..., 0])
            image = np.concatenate((images, images_aug), axis=0)
            mask = np.concatenate((masks, masks_aug), axis=0)
        else:
            image = images.copy()
            mask = masks.copy()
        if not flair: image = image[..., 1:2].copy()
        if not t1: image = image[..., 0:1].copy()
        history = model.fit(image, mask, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        current_epoch += 1
        if history.history['loss'][-1] > 0.99:
            model = get_unet_3(img_shape, first5)
            current_epoch = 1
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path += 'Full_'
    model_path += ''
    if flair: model_path += 'Flair_'
    if t1: model_path += 'T1_'
    if first5: model_path += '5_'
    else: model_path += '3_'
    if aug: model_path += 'Augmentation/'
    else: model_path += 'No_Augmentation/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path += str('_85_15_unet3') + '.h5'
    model.save_weights(model_path)
    print('Model saved to ', model_path)

def train_all_85_15_unet_3_flair(images, masks, flair=True, t1=True, first5=True, aug=True, verbose=False):
    images_delete = [5,7,19,25,26,32] # actual images will be 5, 8, 21, 28, 30, 37
    images_delete_2 = [41,50,51] # 47, 57, 58
    for test_1 in images_delete:
        images = np.delete(images, range(test_1*38, (test_1+1)*38), axis=0)
        masks = np.delete(masks, range(test_1*38, (test_1+1)*38), axis=0)
    for test_2 in images_delete_2:
        images = np.delete(images, range(1520+(test_2-40)*63, 1520+(test_2-39)*63), axis=0)
        masks = np.delete(masks, range(1520+(test_2-40)*63, 1520+(test_2-39)*63), axis=0)
    if aug:
        images = np.concatenate((images, images[..., ::-1, :]), axis=0)
        masks = np.concatenate((masks, masks[..., ::-1, :]), axis=0)
    samples_num = images.shape[0]
    print(samples_num)
    row = images.shape[1]
    col = images.shape[2]
    if aug: epoch = 50
    else: epoch = 200
    batch_size = 30

    img_shape = (row, col, 1)
    model = get_unet_3(img_shape, first5)
    current_epoch = 1
    while current_epoch <= epoch:
        print('Epoch ', str(current_epoch), '/', str(epoch))
        if aug:
            images_aug = np.zeros(images.shape, dtype=np.float32)
            masks_aug = np.zeros(masks.shape, dtype=np.float32)
            for i in range(samples_num):
                images_aug[i, ..., 0], masks_aug[i, ..., 0] = augmentation_2(images[i, ..., 0], masks[i, ..., 0])
            image = np.concatenate((images, images_aug), axis=0)
            mask = np.concatenate((masks, masks_aug), axis=0)
        else:
            image = images.copy()
            mask = masks.copy()
        image = image[..., 0:1].copy()
        history = model.fit(image, mask, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        current_epoch += 1
        if history.history['loss'][-1] > 0.99:
            model = get_unet_3(img_shape, first5)
            current_epoch = 1
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if first5: model_path += '5_'
    else: model_path += '3_'
    if aug: model_path += 'Augmentation/'
    else: model_path += 'No_Augmentation/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path += '_complete_'
    model_path += '_onlyFlair_unet3' + '.h5'
    model.save_weights(model_path)
    print('Model saved to ', model_path)

def train_all_85_15_unet_3_softmax(images, masks, flair=True, t1=True, first5=True, aug=True, verbose=False):
    images_delete = [5,7,19,25,26,32] # actual images will be 5, 8, 21, 28, 30, 37
    images_delete_2 = [41,50,51] # 47, 57, 58
    for test_1 in images_delete:
        images = np.delete(images, range(test_1*38, (test_1+1)*38), axis=0)
        masks = np.delete(masks, range(test_1*38, (test_1+1)*38), axis=0)
    for test_2 in images_delete_2:
        images = np.delete(images, range(1520+(test_2-40)*63, 1520+(test_2-39)*63), axis=0)
        masks = np.delete(masks, range(1520+(test_2-40)*63, 1520+(test_2-39)*63), axis=0)
    if aug:
        images = np.concatenate((images, images[..., ::-1, :]), axis=0)
        masks = np.concatenate((masks, masks[..., ::-1, :]), axis=0)
    samples_num = images.shape[0]
    print(samples_num)
    row = images.shape[1]
    col = images.shape[2]
    if aug: epoch = 50
    else: epoch = 200
    batch_size = 30

    img_shape = (row, col, flair+t1)
    model = get_unet_3_softmax(img_shape, first5)
    current_epoch = 1
    while current_epoch <= epoch:
        print('Epoch ', str(current_epoch), '/', str(epoch))
        if aug:
            images_aug = np.zeros(images.shape, dtype=np.float32)
            masks_aug = np.zeros(masks.shape, dtype=np.float32)
            for i in range(samples_num):
                images_aug[i, ..., 0], images_aug[i, ..., 1], masks_aug[i, ..., 0] = augmentation(images[i, ..., 0], images[i, ..., 1], masks[i, ..., 0])
            image = np.concatenate((images, images_aug), axis=0)
            mask = np.concatenate((masks, masks_aug), axis=0)
        else:
            image = images.copy()
            mask = masks.copy()
        if not flair: image = image[..., 1:2].copy()
        if not t1: image = image[..., 0:1].copy()
        history = model.fit(image, mask, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        current_epoch += 1
        if history.history['loss'][-1] > 0.99:
            model = get_unet_3_softmax(img_shape, first5)
            current_epoch = 1
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path += 'Full_'
    model_path += ''
    if flair: model_path += 'Flair_'
    if t1: model_path += 'T1_'
    if first5: model_path += '5_'
    else: model_path += '3_'
    if aug: model_path += 'Augmentation/'
    else: model_path += 'No_Augmentation/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path += str('_85_15_unet3_softmax') + '.h5'
    model.save_weights(model_path)
    print('Model saved to ', model_path)

# intend with original data (cannot without masks)
def let_see_1():
    patient_num = 60
    array_of_images = np.zeros((4,), dtype=float)
    array_of_masks = np.zeros((4,), dtype=float)
    for patient in range(0,patient_num):
        if patient < 20:
            dir = '../../Data/originalData/Utrecht/'
            name = '_Utrecht_'
        elif patient < 40:
            dir = '../../Data/originalData/Singapore/'
            name = '_Singapore_'
        else:
            dir = '../../Data/originalData/GE3T/'
            name = '_GE3T_'
        dirs = os.listdir(dir)
        dirs.sort()
        dir += dirs[patient%20]
        FLAIR_image = sitk.ReadImage(dir + '/pre/FLAIR.nii.gz')
        T1_image = sitk.ReadImage(dir + '/pre/T1.nii.gz')
        FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
        T1_array = sitk.GetArrayFromImage(T1_image)
        if patient < 40: imgs_test = Utrecht_preprocessing(FLAIR_array, T1_array)
        else: imgs_test = GE3T_preprocessing(FLAIR_array, T1_array)
        print imgs_test.shape

        mask_path = '../../Data/masks/masks_preprocessed/' + str(patient) + name + '.nii.gz'
        mask_image = sitk.ReadImage(mask_path)
        mask_image_array = sitk.GetArrayFromImage(mask_image)
        print mask_image_array.shape
        np.concatenate((array_of_masks, mask_image_array), axis=0)

# train normal with 85 15 proportion
def let_see_2():
    patient_num = 60
    array_of_images = np.zeros((0,0,0), dtype=float)
    array_of_masks = np.zeros((0,0,0), dtype=float)
    for patient in range(0,patient_num):
        if patient < 20:
            name = '_Utrecht_'
        elif patient < 40:
            name = '_Singapore_'
        else:
            name = '_GE3T_'
    
        image_path_flair_path = '../../Data/flair_pre/' + str(patient) + name + '.nii.gz'
        image_path_flair = sitk.ReadImage(image_path_flair_path)
        image_path_flair_array = sitk.GetArrayFromImage(image_path_flair)
        

        image_path_flair_array = image_path_flair_array[..., np.newaxis]

        image_path_t1_path = '../../Data/t1_pre/' + str(patient) + name + '.nii.gz'
        image_path_t1 = sitk.ReadImage(image_path_t1_path)
        image_path_t1_array = sitk.GetArrayFromImage(image_path_t1)
        

        image_path_t1_array = image_path_t1_array[..., np.newaxis]

        image_path_array = np.concatenate([image_path_flair_array, image_path_t1_array], axis=3)
        print image_path_array.shape

        mask_path = '../../Data/masks/masks_preprocessed/' + str(patient) + name + '.nii.gz'
        mask_image = sitk.ReadImage(mask_path)
        mask_image_array = sitk.GetArrayFromImage(mask_image)
        mask_image_array = mask_image_array[..., np.newaxis]
        print mask_image_array.shape

        if patient == 0:
            array_of_images = image_path_array
            array_of_masks = mask_image_array
        else:
            array_of_images = np.concatenate((array_of_images, image_path_array), axis=0)
            array_of_masks = np.concatenate((array_of_masks, mask_image_array), axis=0)
        # np.concatenate((array_of_masks, mask_image_array), axis=0)
    print 'holi'
    print array_of_images.shape
    print array_of_masks.shape
    train_all_85_15(array_of_images, array_of_masks)

# train with patient 0 with mult data
def let_see_3():
    patient_num = 60
    array_of_images = np.zeros((0,0,0), dtype=float)
    array_of_masks = np.zeros((0,0,0), dtype=float)
    for patient in range(0,patient_num):
        if patient < 20:
            name = '_Utrecht_'
        elif patient < 40:
            name = '_Singapore_'
        else:
            name = '_GE3T_'

        image_path = '../../Data/multiplication/pre/pre' + name + str(patient) + '.nii.gz'
        image_path_image = sitk.ReadImage(image_path)
        image_path_array = sitk.GetArrayFromImage(image_path_image)
        

        image_path_array = image_path_array[..., np.newaxis]


        mask_path = '../../Data/masks/masks_preprocessed/' + str(patient) + name + '.nii.gz'
        mask_image = sitk.ReadImage(mask_path)
        mask_image_array = sitk.GetArrayFromImage(mask_image)
        mask_image_array = mask_image_array[..., np.newaxis]
        print mask_image_array.shape

        if patient == 0:
            array_of_images = image_path_array
            array_of_masks = mask_image_array
        else:
            array_of_images = np.concatenate((array_of_images, image_path_array), axis=0)
            array_of_masks = np.concatenate((array_of_masks, mask_image_array), axis=0)
    
    print 'holi'
    print array_of_images.shape
    print array_of_masks.shape
        
    train_only_mult_data(array_of_images, array_of_masks, patient=0)

# train 85 15 with mult data
def let_see_4():
    patient_num = 60
    array_of_images = np.zeros((0,0,0), dtype=float)
    array_of_masks = np.zeros((0,0,0), dtype=float)
    for patient in range(0,patient_num):
        if patient < 20:
            name = '_Utrecht_'
        elif patient < 40:
            name = '_Singapore_'
        else:
            name = '_GE3T_'

        image_path = '../../Data/multiplication/pre/pre' + name + str(patient) + '.nii.gz'
        image_path_image = sitk.ReadImage(image_path)
        image_path_array = sitk.GetArrayFromImage(image_path_image)
        

        image_path_array = image_path_array[..., np.newaxis]


        mask_path = '../../Data/masks/masks_preprocessed/' + str(patient) + name + '.nii.gz'
        mask_image = sitk.ReadImage(mask_path)
        mask_image_array = sitk.GetArrayFromImage(mask_image)
        mask_image_array = mask_image_array[..., np.newaxis]
        print mask_image_array.shape

        if patient == 0:
            array_of_images = image_path_array
            array_of_masks = mask_image_array
        else:
            array_of_images = np.concatenate((array_of_images, image_path_array), axis=0)
            array_of_masks = np.concatenate((array_of_masks, mask_image_array), axis=0)
    
    print 'holi'
    print array_of_images.shape
    print array_of_masks.shape
        
    train_only_mult_data_85_15(array_of_images, array_of_masks)

# train only flair data
def let_see_5():
    patient_num = 60
    array_of_images = np.zeros((0,0,0), dtype=float)
    array_of_masks = np.zeros((0,0,0), dtype=float)
    for patient in range(0,patient_num):
        if patient < 20:
            name = '_Utrecht_'
        elif patient < 40:
            name = '_Singapore_'
        else:
            name = '_GE3T_'
    
        image_path_flair_path = '../../Data/flair_pre/' + str(patient) + name + '.nii.gz'
        image_path_flair = sitk.ReadImage(image_path_flair_path)
        image_path_flair_array = sitk.GetArrayFromImage(image_path_flair)
        

        image_path_flair_array = image_path_flair_array[..., np.newaxis]


        image_path_array = image_path_flair_array
        print image_path_array.shape

        mask_path = '../../Data/masks/masks_preprocessed/' + str(patient) + name + '.nii.gz'
        mask_image = sitk.ReadImage(mask_path)
        mask_image_array = sitk.GetArrayFromImage(mask_image)
        mask_image_array = mask_image_array[..., np.newaxis]
        print mask_image_array.shape

        if patient == 0:
            array_of_images = image_path_array
            array_of_masks = mask_image_array
        else:
            array_of_images = np.concatenate((array_of_images, image_path_array), axis=0)
            array_of_masks = np.concatenate((array_of_masks, mask_image_array), axis=0)
        # np.concatenate((array_of_masks, mask_image_array), axis=0)
    print 'holi'
    print array_of_images.shape
    print array_of_masks.shape
    train_only_85_15_flair(array_of_images, array_of_masks)

# train with 85 15 with unet_2
def let_see_6():
    patient_num = 60
    array_of_images = np.zeros((0,0,0), dtype=float)
    array_of_masks = np.zeros((0,0,0), dtype=float)
    for patient in range(0,patient_num):
        if patient < 20:
            name = '_Utrecht_'
        elif patient < 40:
            name = '_Singapore_'
        else:
            name = '_GE3T_'
    
        image_path_flair_path = '../../Data/flair_pre/' + str(patient) + name + '.nii.gz'
        image_path_flair = sitk.ReadImage(image_path_flair_path)
        image_path_flair_array = sitk.GetArrayFromImage(image_path_flair)
        

        image_path_flair_array = image_path_flair_array[..., np.newaxis]

        image_path_t1_path = '../../Data/t1_pre/' + str(patient) + name + '.nii.gz'
        image_path_t1 = sitk.ReadImage(image_path_t1_path)
        image_path_t1_array = sitk.GetArrayFromImage(image_path_t1)
        

        image_path_t1_array = image_path_t1_array[..., np.newaxis]

        image_path_array = np.concatenate([image_path_flair_array, image_path_t1_array], axis=3)
        print image_path_array.shape

        mask_path = '../../Data/masks/masks_preprocessed/' + str(patient) + name + '.nii.gz'
        mask_image = sitk.ReadImage(mask_path)
        mask_image_array = sitk.GetArrayFromImage(mask_image)
        mask_image_array = mask_image_array[..., np.newaxis]
        print mask_image_array.shape

        if patient == 0:
            array_of_images = image_path_array
            array_of_masks = mask_image_array
        else:
            array_of_images = np.concatenate((array_of_images, image_path_array), axis=0)
            array_of_masks = np.concatenate((array_of_masks, mask_image_array), axis=0)
        # np.concatenate((array_of_masks, mask_image_array), axis=0)
    print array_of_images.shape
    print array_of_masks.shape
    train_all_85_15_unet_2(array_of_images, array_of_masks)

# train with 85 15 with unet_2 pnly flair data
def let_see_7():
    patient_num = 60
    array_of_images = np.zeros((0,0,0), dtype=float)
    array_of_masks = np.zeros((0,0,0), dtype=float)
    for patient in range(0,patient_num):
        if patient < 20:
            name = '_Utrecht_'
        elif patient < 40:
            name = '_Singapore_'
        else:
            name = '_GE3T_'
    
        image_path_flair_path = '../../Data/flair_pre/' + str(patient) + name + '.nii.gz'
        image_path_flair = sitk.ReadImage(image_path_flair_path)
        image_path_flair_array = sitk.GetArrayFromImage(image_path_flair)
        

        image_path_flair_array = image_path_flair_array[..., np.newaxis]


        image_path_array = image_path_flair_array 
        print image_path_array.shape

        mask_path = '../../Data/masks/masks_preprocessed/' + str(patient) + name + '.nii.gz'
        mask_image = sitk.ReadImage(mask_path)
        mask_image_array = sitk.GetArrayFromImage(mask_image)
        mask_image_array = mask_image_array[..., np.newaxis]
        print mask_image_array.shape

        if patient == 0:
            array_of_images = image_path_array
            array_of_masks = mask_image_array
        else:
            array_of_images = np.concatenate((array_of_images, image_path_array), axis=0)
            array_of_masks = np.concatenate((array_of_masks, mask_image_array), axis=0)
        # np.concatenate((array_of_masks, mask_image_array), axis=0)
    print array_of_images.shape
    print array_of_masks.shape
    train_all_85_15_unet_2_flair(array_of_images, array_of_masks)

# train with 85 15 with unet_3
def let_see_8():
    patient_num = 60
    array_of_images = np.zeros((0,0,0), dtype=float)
    array_of_masks = np.zeros((0,0,0), dtype=float)
    for patient in range(0,patient_num):
        if patient < 20:
            name = '_Utrecht_'
        elif patient < 40:
            name = '_Singapore_'
        else:
            name = '_GE3T_'
    
        image_path_flair_path = '../../Data/flair_pre/' + str(patient) + name + '.nii.gz'
        image_path_flair = sitk.ReadImage(image_path_flair_path)
        image_path_flair_array = sitk.GetArrayFromImage(image_path_flair)
        

        image_path_flair_array = image_path_flair_array[..., np.newaxis]

        image_path_t1_path = '../../Data/t1_pre/' + str(patient) + name + '.nii.gz'
        image_path_t1 = sitk.ReadImage(image_path_t1_path)
        image_path_t1_array = sitk.GetArrayFromImage(image_path_t1)
        

        image_path_t1_array = image_path_t1_array[..., np.newaxis]

        image_path_array = np.concatenate([image_path_flair_array, image_path_t1_array], axis=3)
        print image_path_array.shape

        mask_path = '../../Data/masks/masks_preprocessed/' + str(patient) + name + '.nii.gz'
        mask_image = sitk.ReadImage(mask_path)
        mask_image_array = sitk.GetArrayFromImage(mask_image)
        mask_image_array = mask_image_array[..., np.newaxis]
        print mask_image_array.shape

        if patient == 0:
            array_of_images = image_path_array
            array_of_masks = mask_image_array
        else:
            array_of_images = np.concatenate((array_of_images, image_path_array), axis=0)
            array_of_masks = np.concatenate((array_of_masks, mask_image_array), axis=0)
        # np.concatenate((array_of_masks, mask_image_array), axis=0)
    print array_of_images.shape
    print array_of_masks.shape
    train_all_85_15_unet_3(array_of_images, array_of_masks)

# train with 85 15 with unet_3 only flair data
def let_see_9():
    patient_num = 60
    array_of_images = np.zeros((0,0,0), dtype=float)
    array_of_masks = np.zeros((0,0,0), dtype=float)
    for patient in range(0,patient_num):
        if patient < 20:
            name = '_Utrecht_'
        elif patient < 40:
            name = '_Singapore_'
        else:
            name = '_GE3T_'
    
        image_path_flair_path = '../../Data/flair_pre/' + str(patient) + name + '.nii.gz'
        image_path_flair = sitk.ReadImage(image_path_flair_path)
        image_path_flair_array = sitk.GetArrayFromImage(image_path_flair)
        

        image_path_flair_array = image_path_flair_array[..., np.newaxis]


        image_path_array = image_path_flair_array 
        print image_path_array.shape

        mask_path = '../../Data/masks/masks_preprocessed/' + str(patient) + name + '.nii.gz'
        mask_image = sitk.ReadImage(mask_path)
        mask_image_array = sitk.GetArrayFromImage(mask_image)
        mask_image_array = mask_image_array[..., np.newaxis]
        print mask_image_array.shape

        if patient == 0:
            array_of_images = image_path_array
            array_of_masks = mask_image_array
        else:
            array_of_images = np.concatenate((array_of_images, image_path_array), axis=0)
            array_of_masks = np.concatenate((array_of_masks, mask_image_array), axis=0)
        # np.concatenate((array_of_masks, mask_image_array), axis=0)
    print array_of_images.shape
    print array_of_masks.shape
    train_all_85_15_unet_3_flair(array_of_images, array_of_masks)


def let_see_10():
    patient_num = 60
    array_of_images = np.zeros((0,0,0), dtype=float)
    array_of_masks = np.zeros((0,0,0), dtype=float)
    for patient in range(0,patient_num):
        if patient < 20:
            name = '_Utrecht_'
        elif patient < 40:
            name = '_Singapore_'
        else:
            name = '_GE3T_'
    
        image_path_flair_path = '../../Data/flair_pre/' + str(patient) + name + '.nii.gz'
        image_path_flair = sitk.ReadImage(image_path_flair_path)
        image_path_flair_array = sitk.GetArrayFromImage(image_path_flair)
        

        image_path_flair_array = image_path_flair_array[..., np.newaxis]

        image_path_t1_path = '../../Data/t1_pre/' + str(patient) + name + '.nii.gz'
        image_path_t1 = sitk.ReadImage(image_path_t1_path)
        image_path_t1_array = sitk.GetArrayFromImage(image_path_t1)
        

        image_path_t1_array = image_path_t1_array[..., np.newaxis]

        image_path_array = np.concatenate([image_path_flair_array, image_path_t1_array], axis=3)
        print image_path_array.shape

        mask_path = '../../Data/masks/masks_preprocessed/' + str(patient) + name + '.nii.gz'
        mask_image = sitk.ReadImage(mask_path)
        mask_image_array = sitk.GetArrayFromImage(mask_image)
        mask_image_array = mask_image_array[..., np.newaxis]
        print mask_image_array.shape

        if patient == 0:
            array_of_images = image_path_array
            array_of_masks = mask_image_array
        else:
            array_of_images = np.concatenate((array_of_images, image_path_array), axis=0)
            array_of_masks = np.concatenate((array_of_masks, mask_image_array), axis=0)
        # np.concatenate((array_of_masks, mask_image_array), axis=0)
    print array_of_images.shape
    print array_of_masks.shape
    train_all_85_15_unet_3_softmax(array_of_images, array_of_masks)
# train with numpy arrays

def let_see_0():
    warnings.filterwarnings("ignore")
    images = np.load('data/images_three_datasets_sorted.npy')
    print(images.shape)
    masks = np.load('data/masks_three_datasets_sorted.npy')
    print(masks.shape)
    patient_num  = 60
    for patient in range(0, patient_num):
        train_leave_one_out(images, masks, patient=patient, full=True, verbose=True)
#leave-one-out evaluation
def main():
    let_see_9()

if __name__=='__main__':
    main()
