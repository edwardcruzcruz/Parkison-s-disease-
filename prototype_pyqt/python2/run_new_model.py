# -*- coding: utf-8 -*-

import os
import time
import scipy
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, BatchNormalization, Activation
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import apply_transform, transform_matrix_offset_center
import SimpleITK as sitk
import warnings
K.set_image_data_format('channels_last')
import unicodedata
import sqlite3
import datetime

smooth=1.
rows_standard = 200
cols_standard = 200
thresh_FLAIR = 70      #to mask the brain


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

def Utrecht_preprocessing_flair(FLAIR_image):
    FLAIR_image_2 = FLAIR_image.copy()
    try:
        num_selected_slice = np.shape(FLAIR_image)[0]
        image_rows_Dataset = np.shape(FLAIR_image)[1]
        image_cols_Dataset = np.shape(FLAIR_image)[2]

        brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
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

    except Exception:
        print("something went wrong")
    finally:
        FLAIR_image_2  = FLAIR_image_2 [..., np.newaxis]
        return FLAIR_image_2 

def GE3T_preprocessing_flair(FLAIR_image):
    FLAIR_image_2 = FLAIR_image.copy()
    try:
        start_cut = 46
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

        FLAIR_image_suitable[...] = np.min(FLAIR_image)
        FLAIR_image_suitable[:, :, (cols_standard/2-image_cols_Dataset/2):(cols_standard/2+image_cols_Dataset/2)] = FLAIR_image[:, start_cut:start_cut+rows_standard, :]
    
        FLAIR_image_suitable  = FLAIR_image_suitable[..., np.newaxis]
        return FLAIR_image_suitable
    except Exception:
        print("something went wrong")
    finally:
        FLAIR_image_2  = FLAIR_image_2 [..., np.newaxis]
        return FLAIR_image_2 

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

def train_net_flair(images, masks, name_model, first5=True, aug=True, verbose=False):
    if aug:
        images = np.concatenate((images, images[..., ::-1, :]), axis=0)
        masks = np.concatenate((masks, masks[..., ::-1, :]), axis=0)
    samples_num = images.shape[0]
    row = images.shape[1]
    col = images.shape[2]
    if aug: epoch = 50
    else: epoch = 200
    batch_size = 30

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
        history = model.fit(image, mask, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        current_epoch += 1
        if history.history['loss'][-1] > 0.99:
            model = get_unet(img_shape, first5)
            current_epoch = 1
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    name = ( name_model + "_" + str(datetime.datetime.now()) + "_" + str(rows_standard) + "_" + strc(cols_standard)).strip()
    model_path += name + '.h5'
    model.save_weights(model_path)
    print('Model saved to ', model_path)
    save_new_model(name, model_path)
    

def save_new_model(name, model_path):
    con = sql_connection()
    cursor = con.cursor()
    cursor.execute("SELECT * FROM model ORDER BY id DESC LIMIT 1;")
    lastModel = cursor.fetchone()
    last_index = lastModel[0]
    sql_insert = """
            UPDATE model
            SET name = ? ,
                path = ? ,
                executionDate = ?
            WHERE id = ?
    """
    execution_date = datetime.datetime.now()
    update_tuple = (name.decode('utf-8'), model_path.decode('utf-8'), execution_date, last_index)
    cursor.execute(sql_insert, update_tuple)
    con.close()




def sql_connection():
    try:
        con = sqlite3.connect('theDatabase.db')
        return con
    except Error:
        print Error

def get_model_images_sqlite():
    global rows_standard
    global cols_standard
    con = sql_connection()
    cursor = con.cursor()
    cursor.execute("SELECT * FROM model ORDER BY id DESC LIMIT 1;")
    lastModel = cursor.fetchone()
    name = lastModel[1]
    rows_standard = lastModel[2]
    cols_standard = lastModel[3]
    arrayImagesPath = []
    cursor.execute("SELECT * FROM image")
    allImages = cursor.fetchall()
    for image in allImages:
        imagePath = image[2]
        maskPath = image[3]
        tuplePath = (imagePath, maskPath)
        arrayImagesPath.append(tuplePath)
    
    con.close()

    return name, arrayImagesPath

def verify_name_model(name_model):
    name_model = name_model.encode('ascii', 'ignore')
    name_model_split = name_model.split("_")
    if len(name_model_split) > 1:
        return name_model_split[0]
    return name_model

def train_model_in_background(name_model, pathImages):
    array_of_images_38 = np.zeros((0,0,0), dtype=float)
    array_of_images_63 = np.zeros((0,0,0), dtype=float)

    array_of_masks_38 = np.zeros((0,0,0), dtype=float)
    array_of_masks_63 = np.zeros((0,0,0), dtype=float)
    counter_of_38 = 0
    counter_of_63 = 0

    new_name_model = verify_name_model(name_model)

    for patient_path in pathImages:
        image_path = patient_path[0]
        mask_path = patient_path[1]

        image_path_flair = sitk.ReadImage(image_path.encode('ascii','ignore'))
        image_path_flair_array = sitk.GetArrayFromImage(image_path_flair)
        #image_path_flair_array = image_path_flair_array[..., np.newaxis]

        mask_path_ = sitk.ReadImage(mask_path.encode('ascii','ignore'))
        mask_path_array = sitk.GetArrayFromImage(mask_path_)
        #mask_path_array = mask_path_array [..., np.newaxis]



        if image_path_flair_array.shape[0] == 38:
            image_path_flair_array = Utrecht_preprocessing_flair(image_path_flair_array)
            mask_path_array = Utrecht_preprocessing_flair(mask_path_array)
            if counter_of_38 == 0:
                array_of_images_38 = image_path_flair_array
                array_of_masks_38  = mask_path_array
            else:
                array_of_images_38 = np.concatenate((array_of_images_38, image_path_flair_array), axis=0)
                array_of_masks_38  = np.concatenate((array_of_masks_38, mask_path_array), axis=0)
            counter_of_38 +=1

        elif image_path_flair_array.shape[0] == 63:
            image_path_flair_array = GE3T_preprocessing_flair(image_path_flair_array)
            mask_path_array = Utrecht_preprocessing_flair(mask_path_array)
            if counter_of_63 == 0:
                array_of_images_63 = image_path_flair_array
                array_of_masks_63  = mask_path_array
            else:
                array_of_images_63 = np.concatenate((array_of_images_63, image_path_flair_array), axis=0)
                array_of_masks_63  = np.concatenate((array_of_masks_63, mask_path_array), axis=0)
            counter_of_63 +=1
    

    array_of_images = np.concatenate((array_of_images_38, array_of_images_63), axis=0)
    array_of_masks = np.concatenate((array_of_masks_38, array_of_masks_63), axis=0)

    train_net_flair(array_of_images, array_of_masks, new_name_model)

if __name__=='__main__':
    name, arrayImagesPath = get_model_images_sqlite()
    train_model_in_background(name, arrayImagesPath)


