import os
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
from evaluation import getDSC, getHausdorff, getLesionDetection, getAVD, getImages
K.set_image_data_format('channels_last')

rows_standard = 200
cols_standard = 200
thresh_FLAIR = 70      #to mask the brain
thresh_T1 = 30
smooth=1.

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

def Utrecht_preprocessing_flair(FLAIR_image):
    channel_num = 2
    #print(np.shape(FLAIR_image))
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
    FLAIR_image  = FLAIR_image[..., np.newaxis]
    #print(np.shape(imgs_two_channels))
    return FLAIR_image

def GE3T_preprocessing_flair(FLAIR_image):
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

def Utrecht_postprocessing(FLAIR_array, pred):
    start_slice = 6
    num_selected_slice = np.shape(FLAIR_array)[0]
    image_rows_Dataset = np.shape(FLAIR_array)[1]
    image_cols_Dataset = np.shape(FLAIR_array)[2]
    original_pred = np.ndarray(np.shape(FLAIR_array), dtype=np.float32)
    original_pred[...] = 0
    
    original_pred[:,(image_rows_Dataset-rows_standard)/2:(image_rows_Dataset+rows_standard)/2,(image_cols_Dataset-cols_standard)/2:(image_cols_Dataset+cols_standard)/2] = pred[:,:,:,0]
    
    original_pred[0:start_slice, :, :] = 0
    original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
    return original_pred

def GE3T_preprocessing(FLAIR_image, T1_image):

    channel_num = 2
    start_cut = 46
    num_selected_slice = np.shape(FLAIR_image)[0]
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

def GE3T_postprocessing(FLAIR_array, pred):
    start_slice = 11
    start_cut = 46
    num_selected_slice = np.shape(FLAIR_array)[0]
    image_rows_Dataset = np.shape(FLAIR_array)[1]
    image_cols_Dataset = np.shape(FLAIR_array)[2]
    original_pred = np.ndarray(np.shape(FLAIR_array), dtype=np.float32)
    original_pred[...] = 0
    original_pred[:, start_cut:start_cut+rows_standard,:] = pred[:,:, (rows_standard-image_cols_Dataset)/2:(rows_standard+image_cols_Dataset)/2,0]

    original_pred[0:start_slice, :, :] = 0
    original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
    print(original_pred)
    return original_pred

# for patient 0 and common data
def test_leave_one_out_common(patient=0, flair=True, t1=True, full=True, first5=True, aug=True, verbose=False):
    if patient < 20: dir = '../../Data/originalData/Utrecht/'
    elif patient < 40: dir = '../../Data/originalData/Singapore/'
    else: dir = '../../Data/originalData/GE3T/'
    dirs = os.listdir(dir)
    dirs.sort()
    dir += dirs[patient%20]
    FLAIR_image = sitk.ReadImage(dir + '/pre/FLAIR.nii.gz')
    T1_image = sitk.ReadImage(dir + '/pre/T1.nii.gz')
    FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
    T1_array = sitk.GetArrayFromImage(T1_image)
    if patient < 40: imgs_test = Utrecht_preprocessing(FLAIR_array, T1_array)
    else: imgs_test = GE3T_preprocessing(FLAIR_array, T1_array)
    if not flair: imgs_test = imgs_test[..., 1:2].copy()
    if not t1: imgs_test = imgs_test[..., 0:1].copy()
    img_shape = (rows_standard, cols_standard, flair+t1)
    model = get_unet(img_shape, first5)
    model_path = 'models/Full_Flair_T1_5_Augmentation/'
    #if you want to test three models ensemble, just do like this: pred = (pred_1+pred_2+pred_3)/3
    model.load_weights(model_path + str(patient) + '.h5')
    pred = model.predict(imgs_test, batch_size=1, verbose=verbose)
    pred[pred > 0.5] = 1.
    pred[pred <= 0.5] = 0.
    if patient < 40: original_pred = Utrecht_postprocessing(FLAIR_array, pred)
    else: original_pred = GE3T_postprocessing(FLAIR_array, pred)
    filename_resultImage = model_path + str(patient) + '.nii.gz'
    sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage )
    filename_testImage = os.path.join(dir + '/wmh.nii.gz')
    testImage, resultImage = getImages(filename_testImage, filename_resultImage)
    dsc = getDSC(testImage, resultImage)
    avd = getAVD(testImage, resultImage) 
    h95 = getHausdorff(testImage, resultImage)
    recall, f1 = getLesionDetection(testImage, resultImage)
    return dsc, h95, avd, recall, f1

# for all test patients
def test_leave_one_out_all(flair=True, t1=True, full=True, first5=True, aug=True, verbose=False):
    theImages = [5,8,21,28,30,37,47,57,58] # actual images will be 5, 8, 21, 28, 30, 37
    theDict = {}
    for patient in theImages:
        if patient < 20: dir = '../../Data/originalData/Utrecht/'
        elif patient < 40: dir = '../../Data/originalData/Singapore/'
        else: dir = '../../Data/originalData/GE3T/'
        dirs = os.listdir(dir)
        dirs.sort()
        dir += dirs[patient%20]
        FLAIR_image = sitk.ReadImage(dir + '/pre/FLAIR.nii.gz')
        T1_image = sitk.ReadImage(dir + '/pre/T1.nii.gz')
        FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
        T1_array = sitk.GetArrayFromImage(T1_image)
        if patient < 40: imgs_test = Utrecht_preprocessing(FLAIR_array, T1_array)
        else: imgs_test = GE3T_preprocessing(FLAIR_array, T1_array)
        if not flair: imgs_test = imgs_test[..., 1:2].copy()
        if not t1: imgs_test = imgs_test[..., 0:1].copy()
        img_shape = (rows_standard, cols_standard, flair+t1)
        model = get_unet(img_shape, first5)
        model_path = 'models/Full_Flair_T1_5_Augmentation/'
        #if you want to test three models ensemble, just do like this: pred = (pred_1+pred_2+pred_3)/3
        model.load_weights(model_path + '_85_15.h5')
        pred = model.predict(imgs_test, batch_size=1, verbose=verbose)
        pred[pred > 0.5] = 1.
        pred[pred <= 0.5] = 0.
        if patient < 40: original_pred = Utrecht_postprocessing(FLAIR_array, pred)
        else: original_pred = GE3T_postprocessing(FLAIR_array, pred)
        filename_resultImage = model_path + str(patient) + '.nii.gz'
        sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage )
        filename_testImage = os.path.join(dir + '/wmh.nii.gz')
        testImage, resultImage = getImages(filename_testImage, filename_resultImage)
        dsc = getDSC(testImage, resultImage)
        avd = getAVD(testImage, resultImage) 
        h95 = getHausdorff(testImage, resultImage)
        recall, f1 = getLesionDetection(testImage, resultImage)
        theDict[patient] = [dsc, h95, avd, recall, f1]
    return theDict

def test_leave_one_out_all_mult(flair=True, t1=True, full=True, first5=True, aug=True, verbose=False):
    theImages = [5,8,21,28,30,37,47,57,58] # actual images will be 5, 8, 21, 28, 30, 37
    theDict = {}
    for patient in theImages:
        if patient < 20: dir = '../../Data/originalData/Utrecht/'
        elif patient < 40: dir = '../../Data/originalData/Singapore/'
        else: dir = '../../Data/originalData/GE3T/'
        dirs = os.listdir(dir)
        dirs.sort()
        dir += dirs[patient%20]
        FLAIR_image = sitk.ReadImage(dir + '/pre/FLAIR.nii.gz')
        T1_image = sitk.ReadImage(dir + '/pre/T1.nii.gz')
        FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
        T1_array = sitk.GetArrayFromImage(T1_image)
        if patient < 40: imgs_test = Utrecht_preprocessing(FLAIR_array, T1_array)
        else: imgs_test = GE3T_preprocessing(FLAIR_array, T1_array)
        flair_image = imgs_test[..., 0:1].copy()
        t1_image = imgs_test = imgs_test[..., 1:2].copy()
        imgs_test = np.multiply(flair_image, t1_image)
        img_shape = (rows_standard, cols_standard, 1)
        model = get_unet(img_shape, first5)
        model_path = 'models/multData/'
        #if you want to test three models ensemble, just do like this: pred = (pred_1+pred_2+pred_3)/3
        model.load_weights(model_path + '_complete__multData.h5')
        pred = model.predict(imgs_test, batch_size=1, verbose=verbose)
        pred[pred > 0.5] = 1.
        pred[pred <= 0.5] = 0.
        if patient < 40: original_pred = Utrecht_postprocessing(FLAIR_array, pred)
        else: original_pred = GE3T_postprocessing(FLAIR_array, pred)
        filename_resultImage = model_path + str(patient) + '.nii.gz'
        sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage )
        filename_testImage = os.path.join(dir + '/wmh.nii.gz')
        testImage, resultImage = getImages(filename_testImage, filename_resultImage)
        dsc = getDSC(testImage, resultImage)
        avd = getAVD(testImage, resultImage) 
        h95 = getHausdorff(testImage, resultImage)
        recall, f1 = getLesionDetection(testImage, resultImage)
        theDict[patient] = [dsc, h95, avd, recall, f1]
    return theDict

def test_leave_one_out_all_flair( first5=True, verbose=False, combinedModel=False):
    theImages = [5,8,21,28,30,37,47,57,58] # actual images will be 5, 8, 21, 28, 30, 37
    theDict = {}
    for patient in theImages:
        if patient < 20: dir = '../../Data/originalData/Utrecht/'
        elif patient < 40: dir = '../../Data/originalData/Singapore/'
        else: dir = '../../Data/originalData/GE3T/'
        dirs = os.listdir(dir)
        dirs.sort()
        dir += dirs[patient%20]
        FLAIR_image = sitk.ReadImage(dir + '/pre/FLAIR.nii.gz')
        FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
        if patient < 40: imgs_test = Utrecht_preprocessing_flair(FLAIR_array)
        else: imgs_test = GE3T_preprocessing_flair(FLAIR_array)
        flair_image = imgs_test[..., 0:1].copy()
        imgs_test = flair_image
        img_shape = (rows_standard, cols_standard, 1)
        model = get_unet(img_shape, first5)
        #if you want to test three models ensemble, just do like this: pred = (pred_1+pred_2+pred_3)/3
        if combinedModel:
            model_path = 'models/Full_Flair_T1_5_Augmentation/'
            model.load_weights(model_path + '_85_15.h5')
        else:
            model_path = 'models/onlyFlair/'
            model.load_weights(model_path + '_complete__onlyFlair.h5')
        pred = model.predict(imgs_test, batch_size=1, verbose=verbose)
        pred[pred > 0.5] = 1.
        pred[pred <= 0.5] = 0.
        if patient < 40: original_pred = Utrecht_postprocessing(FLAIR_array, pred)
        else: original_pred = GE3T_postprocessing(FLAIR_array, pred)
        filename_resultImage = model_path + str(patient) + '.nii.gz'
        sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage )
        filename_testImage = os.path.join(dir + '/wmh.nii.gz')
        testImage, resultImage = getImages(filename_testImage, filename_resultImage)
        dsc = getDSC(testImage, resultImage)
        avd = getAVD(testImage, resultImage) 
        h95 = getHausdorff(testImage, resultImage)
        recall, f1 = getLesionDetection(testImage, resultImage)
        theDict[patient] = [dsc, h95, avd, recall, f1]
    return theDict

def test_leave_one_out_all_unet2(flair=True, t1=True, full=True, first5=True, aug=True, verbose=False):
    theImages = [5,8,21,28,30,37,47,57,58] # actual images will be 5, 8, 21, 28, 30, 37
    theDict = {}
    for patient in theImages:
        if patient < 20: dir = '../../Data/originalData/Utrecht/'
        elif patient < 40: dir = '../../Data/originalData/Singapore/'
        else: dir = '../../Data/originalData/GE3T/'
        dirs = os.listdir(dir)
        dirs.sort()
        dir += dirs[patient%20]
        FLAIR_image = sitk.ReadImage(dir + '/pre/FLAIR.nii.gz')
        T1_image = sitk.ReadImage(dir + '/pre/T1.nii.gz')
        FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
        T1_array = sitk.GetArrayFromImage(T1_image)
        if patient < 40: imgs_test = Utrecht_preprocessing(FLAIR_array, T1_array)
        else: imgs_test = GE3T_preprocessing(FLAIR_array, T1_array)
        if not flair: imgs_test = imgs_test[..., 1:2].copy()
        if not t1: imgs_test = imgs_test[..., 0:1].copy()
        img_shape = (rows_standard, cols_standard, flair+t1)
        model = get_unet_2(img_shape, first5)
        model_path = 'models/unet2/'
        #if you want to test three models ensemble, just do like this: pred = (pred_1+pred_2+pred_3)/3
        model.load_weights(model_path + '_85_15_unet2.h5')
        pred = model.predict(imgs_test, batch_size=1, verbose=verbose)
        pred[pred > 0.5] = 1.
        pred[pred <= 0.5] = 0.
        if patient < 40: original_pred = Utrecht_postprocessing(FLAIR_array, pred)
        else: original_pred = GE3T_postprocessing(FLAIR_array, pred)
        filename_resultImage = model_path + str(patient) + '.nii.gz'
        sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage )
        filename_testImage = os.path.join(dir + '/wmh.nii.gz')
        testImage, resultImage = getImages(filename_testImage, filename_resultImage)
        dsc = getDSC(testImage, resultImage)
        avd = getAVD(testImage, resultImage) 
        h95 = getHausdorff(testImage, resultImage)
        recall, f1 = getLesionDetection(testImage, resultImage)
        theDict[patient] = [dsc, h95, avd, recall, f1]
    return theDict

def test_leave_one_out_all_unet2_flair( first5=True, verbose=False, combinedModel=False):
    theImages = [5,8,21,28,30,37,47,57,58] # actual images will be 5, 8, 21, 28, 30, 37
    theDict = {}
    for patient in theImages:
        if patient < 20: dir = '../../Data/originalData/Utrecht/'
        elif patient < 40: dir = '../../Data/originalData/Singapore/'
        else: dir = '../../Data/originalData/GE3T/'
        dirs = os.listdir(dir)
        dirs.sort()
        dir += dirs[patient%20]
        FLAIR_image = sitk.ReadImage(dir + '/pre/FLAIR.nii.gz')
        FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
        if patient < 40: imgs_test = Utrecht_preprocessing_flair(FLAIR_array)
        else: imgs_test = GE3T_preprocessing_flair(FLAIR_array)
        flair_image = imgs_test[..., 0:1].copy()
        imgs_test = flair_image
        img_shape = (rows_standard, cols_standard, 1)
        model = get_unet_2(img_shape, first5)
        if combinedModel:
            model_path = 'models/unet2/'
            model.load_weights(model_path + '_85_15_unet2.h5')
        else:
            model_path = 'models/unet2_onlyFlair/'
            model.load_weights(model_path + '_complete__onlyFlair_unet2.h5')
        pred = model.predict(imgs_test, batch_size=1, verbose=verbose)
        pred[pred > 0.5] = 1.
        pred[pred <= 0.5] = 0.
        if patient < 40: original_pred = Utrecht_postprocessing(FLAIR_array, pred)
        else: original_pred = GE3T_postprocessing(FLAIR_array, pred)
        filename_resultImage = model_path + str(patient) + '.nii.gz'
        sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage )
        filename_testImage = os.path.join(dir + '/wmh.nii.gz')
        testImage, resultImage = getImages(filename_testImage, filename_resultImage)
        dsc = getDSC(testImage, resultImage)
        avd = getAVD(testImage, resultImage) 
        h95 = getHausdorff(testImage, resultImage)
        recall, f1 = getLesionDetection(testImage, resultImage)
        theDict[patient] = [dsc, h95, avd, recall, f1]
    return theDict

def test_leave_one_out_all_unet3(flair=True, t1=True, full=True, first5=True, aug=True, verbose=False):
    theImages = [5,8,21,28,30,37,47,57,58] # actual images will be 5, 8, 21, 28, 30, 37
    theDict = {}
    for patient in theImages:
        if patient < 20: dir = '../../Data/originalData/Utrecht/'
        elif patient < 40: dir = '../../Data/originalData/Singapore/'
        else: dir = '../../Data/originalData/GE3T/'
        dirs = os.listdir(dir)
        dirs.sort()
        dir += dirs[patient%20]
        FLAIR_image = sitk.ReadImage(dir + '/pre/FLAIR.nii.gz')
        T1_image = sitk.ReadImage(dir + '/pre/T1.nii.gz')
        FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
        T1_array = sitk.GetArrayFromImage(T1_image)
        if patient < 40: imgs_test = Utrecht_preprocessing(FLAIR_array, T1_array)
        else: imgs_test = GE3T_preprocessing(FLAIR_array, T1_array)
        if not flair: imgs_test = imgs_test[..., 1:2].copy()
        if not t1: imgs_test = imgs_test[..., 0:1].copy()
        img_shape = (rows_standard, cols_standard, flair+t1)
        model = get_unet_3(img_shape, first5)
        model_path = 'models/unet3/'
        #if you want to test three models ensemble, just do like this: pred = (pred_1+pred_2+pred_3)/3
        model.load_weights(model_path + '_85_15_unet3.h5')
        pred = model.predict(imgs_test, batch_size=1, verbose=verbose)
        pred[pred > 0.5] = 1.
        pred[pred <= 0.5] = 0.
        if patient < 40: original_pred = Utrecht_postprocessing(FLAIR_array, pred)
        else: original_pred = GE3T_postprocessing(FLAIR_array, pred)
        filename_resultImage = model_path + str(patient) + '.nii.gz'
        sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage )
        filename_testImage = os.path.join(dir + '/wmh.nii.gz')
        testImage, resultImage = getImages(filename_testImage, filename_resultImage)
        dsc = getDSC(testImage, resultImage)
        avd = getAVD(testImage, resultImage) 
        h95 = getHausdorff(testImage, resultImage)
        recall, f1 = getLesionDetection(testImage, resultImage)
        theDict[patient] = [dsc, h95, avd, recall, f1]
    return theDict

def test_leave_one_out_all_unet3_flair( first5=True, verbose=False, combinedModel=False):
    theImages = [5,8,21,28,30,37,47,57,58] # actual images will be 5, 8, 21, 28, 30, 37
    theDict = {}
    for patient in theImages:
        if patient < 20: dir = '../../Data/originalData/Utrecht/'
        elif patient < 40: dir = '../../Data/originalData/Singapore/'
        else: dir = '../../Data/originalData/GE3T/'
        dirs = os.listdir(dir)
        dirs.sort()
        dir += dirs[patient%20]
        FLAIR_image = sitk.ReadImage(dir + '/pre/FLAIR.nii.gz')
        FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
        if patient < 40: imgs_test = Utrecht_preprocessing_flair(FLAIR_array)
        else: imgs_test = GE3T_preprocessing_flair(FLAIR_array)
        flair_image = imgs_test[..., 0:1].copy()
        imgs_test = flair_image
        img_shape = (rows_standard, cols_standard, 1)
        model = get_unet_3(img_shape, first5)
        if combinedModel:
            model_path = 'models/unet3/'
            #if you want to test three models ensemble, just do like this: pred = (pred_1+pred_2+pred_3)/3
            model.load_weights(model_path + '_85_15_unet3.h5')
        else:
            model_path = 'models/unet3_onlyFlair/'
            #if you want to test three models ensemble, just do like this: pred = (pred_1+pred_2+pred_3)/3
            model.load_weights(model_path + '_complete__onlyFlair_unet3.h5')  
        pred = model.predict(imgs_test, batch_size=1, verbose=verbose)
        pred[pred > 0.5] = 1.
        pred[pred <= 0.5] = 0.
        if patient < 40: original_pred = Utrecht_postprocessing(FLAIR_array, pred)
        else: original_pred = GE3T_postprocessing(FLAIR_array, pred)
        filename_resultImage = model_path + str(patient) + '.nii.gz'
        sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage )
        filename_testImage = os.path.join(dir + '/wmh.nii.gz')
        testImage, resultImage = getImages(filename_testImage, filename_resultImage)
        dsc = getDSC(testImage, resultImage)
        avd = getAVD(testImage, resultImage) 
        h95 = getHausdorff(testImage, resultImage)
        recall, f1 = getLesionDetection(testImage, resultImage)
        theDict[patient] = [dsc, h95, avd, recall, f1]
    return theDict

# comparison between all the models with the test_data is the same dimension as the training_data
def comparison(lang="en"):
 
    filePath = "results-en.csv" if lang == "en" else "results-es.csv"
 
    with open(filePath, 'w') as theFile:
        if lang == "en":
            theFile.write('Applied model,patient,metric,value\n')
        else:
            theFile.write('Modelo aplicado,patient,metric,value\n')
        theDict = test_leave_one_out_all() # common 
        for key in theDict:
            dsc = theDict[key][0]
            h95 = theDict[key][1]
            avd = theDict[key][2]
            recall = theDict[key][3]
            f1 = theDict[key][4]
            print('Result of patient ' + str(key))
            print('Dice',                dsc,       '(higher is better, max=1)')
            print('HD',                  h95, 'mm',  '(lower is better, min=0)')
            print('AVD',                 avd,  '%',  '(lower is better, min=0)')
            print('Lesion detection', recall,       '(higher is better, max=1)')
            print('Lesion F1',            f1,       '(higher is better, max=1)')
            if lang == "en":
                theFile.write('State of the art,' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('State of the art,' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('State of the art,' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('State of the art,' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('State of the art,' + str(key) + ',Lesion F1,' + str(f1) + '\n')
            else:
                theFile.write('Estado del arte,' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('Estado del arte,' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('Estado del arte,' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('Estado del arte,' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('Estado del arte,' + str(key) + ',Lesion F1,' + str(f1) + '\n')
    


        theDict2 = test_leave_one_out_all_mult() # mult data
        for key in theDict2:
            dsc = theDict2[key][0]
            h95 = theDict2[key][1]
            avd = theDict2[key][2]
            recall = theDict2[key][3]
            f1 = theDict2[key][4]
            print('Result of patient ' + str(key))
            print('Dice',                dsc,       '(higher is better, max=1)')
            print('HD',                  h95, 'mm',  '(lower is better, min=0)')
            print('AVD',                 avd,  '%',  '(lower is better, min=0)')
            print('Lesion detection', recall,       '(higher is better, max=1)')
            print('Lesion F1',            f1,       '(higher is better, max=1)')
            if lang == "en":
                theFile.write('State of the art (FLAIR X T1),' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('State of the art (FLAIR X T1),' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('State of the art (FLAIR X T1),' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('State of the art (FLAIR X T1),' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('State of the art (FLAIR X T1),' + str(key) + ',Lesion F1,' + str(f1) + '\n')
            else:
                theFile.write('Exp. Datos Multiplicados,' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('Exp. Datos Multiplicados,' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('Exp. Datos Multiplicados,' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('Exp. Datos Multiplicados,' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('Exp. Datos Multiplicados,' + str(key) + ',Lesion F1,' + str(f1) + '\n')
        
        theDict3 = test_leave_one_out_all_flair() # mult data
        for key in theDict3:
            dsc = theDict3[key][0]
            h95 = theDict3[key][1]
            avd = theDict3[key][2]
            recall = theDict3[key][3]
            f1 = theDict3[key][4]
            print('Result of patient ' + str(key))
            print('Dice',                dsc,       '(higher is better, max=1)')
            print('HD',                  h95, 'mm',  '(lower is better, min=0)')
            print('AVD',                 avd,  '%',  '(lower is better, min=0)')
            print('Lesion detection', recall,       '(higher is better, max=1)')
            print('Lesion F1',            f1,       '(higher is better, max=1)')
            if lang == "en":
                theFile.write('State of the art (FLAIR only),' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('State of the art (FLAIR only),' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('State of the art (FLAIR only),' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('State of the art (FLAIR only),' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('State of the art (FLAIR only),' + str(key) + ',Lesion F1,' + str(f1) + '\n')
            else:
                theFile.write('Exp. solo MRI FLAIR,' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('Exp. solo MRI FLAIR,' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('Exp. solo MRI FLAIR,' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('Exp. solo MRI FLAIR,' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('Exp. solo MRI FLAIR,' + str(key) + ',Lesion F1,' + str(f1) + '\n')

    
        theDict4 = test_leave_one_out_all_unet2()
        for key in theDict4:
            dsc = theDict4[key][0]
            h95 = theDict4[key][1]
            avd = theDict4[key][2]
            recall = theDict4[key][3]
            f1 = theDict4[key][4]
            print('Result of patient ' + str(key))
            print('Dice',                dsc,       '(higher is better, max=1)')
            print('HD',                  h95, 'mm',  '(lower is better, min=0)')
            print('AVD',                 avd,  '%',  '(lower is better, min=0)')
            print('Lesion detection', recall,       '(higher is better, max=1)')
            print('Lesion F1',            f1,       '(higher is better, max=1)')
            if lang == "en":
                theFile.write('U-net #2,' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('U-net #2,' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('U-net #2,' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('U-net #2' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('U-net #2,' + str(key) + ',Lesion F1,' + str(f1) + '\n')
            else:
                theFile.write('Exp. UNET menos una capa,' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('Exp. UNET menos una capa,' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('Exp. UNET menos una capa,' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('Exp. UNET menos una capa,' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('Exp. UNET menos una capa,' + str(key) + ',Lesion F1,' + str(f1) + '\n')

        theDict5 = test_leave_one_out_all_unet2_flair()
        for key in theDict5:
            dsc = theDict5[key][0]
            h95 = theDict5[key][1]
            avd = theDict5[key][2]
            recall = theDict5[key][3]
            f1 = theDict5[key][4]
            print('Result of patient ' + str(key))
            print('Dice',                dsc,       '(higher is better, max=1)')
            print('HD',                  h95, 'mm',  '(lower is better, min=0)')
            print('AVD',                 avd,  '%',  '(lower is better, min=0)')
            print('Lesion detection', recall,       '(higher is better, max=1)')
            print('Lesion F1',            f1,       '(higher is better, max=1)')
            if lang == "en":
                theFile.write('U-net #2 (FLAIR only),' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('U-net #2 (FLAIR only),' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('U-net #2 (FLAIR only),' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('U-net #2 (FLAIR only)' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('U-net #2 (FLAIR only),' + str(key) + ',Lesion F1,' + str(f1) + '\n')
            else:
                theFile.write('Exp. UNET menos una capa solo FLAIR,' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('Exp. UNET menos una capa solo FLAIR,' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('Exp. UNET menos una capa solo FLAIR,' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('Exp. UNET menos una capa solo FLAIR,' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('Exp. UNET menos una capa solo FLAIR,' + str(key) + ',Lesion F1,' + str(f1) + '\n')

        theDict6 = test_leave_one_out_all_unet3()
        for key in theDict:
            dsc = theDict6[key][0]
            h95 = theDict6[key][1]
            avd = theDict6[key][2]
            recall = theDict6[key][3]
            f1 = theDict6[key][4]
            print('Result of patient ' + str(key))
            print('Dice',                dsc,       '(higher is better, max=1)')
            print('HD',                  h95, 'mm',  '(lower is better, min=0)')
            print('AVD',                 avd,  '%',  '(lower is better, min=0)')
            print('Lesion detection', recall,       '(higher is better, max=1)')
            print('Lesion F1',            f1,       '(higher is better, max=1)')
            if lang == "en":
                theFile.write('U-net #3,' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('U-net #3,' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('U-net #3,' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('U-net #3' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('U-net #3,' + str(key) + ',Lesion F1,' + str(f1) + '\n')
            
            else:
                theFile.write('Unet con elu y he_normal,' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('Unet con elu y he_normal,' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('Unet con elu y he_normal,' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('Unet con elu y he_normal,' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('Unet con elu y he_normal,' + str(key) + ',Lesion F1,' + str(f1) + '\n')
    
        theDict7 = test_leave_one_out_all_unet3_flair()
        for key in theDict:
            dsc = theDict7[key][0]
            h95 = theDict7[key][1]
            avd = theDict7[key][2]
            recall = theDict7[key][3]
            f1 = theDict7[key][4]
            print('Result of patient ' + str(key))
            print('Dice',                dsc,       '(higher is better, max=1)')
            print('HD',                  h95, 'mm',  '(lower is better, min=0)')
            print('AVD',                 avd,  '%',  '(lower is better, min=0)')
            print('Lesion detection', recall,       '(higher is better, max=1)')
            print('Lesion F1',            f1,       '(higher is better, max=1)')
            if lang == "en":
                theFile.write('U-net #3 (FLAIR only),' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('U-net #3 (FLAIR only),' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('U-net #3 (FLAIR only),' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('U-net #3 (FLAIR only),' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('U-net #3 (FLAIR only),' + str(key) + ',Lesion F1,' + str(f1) + '\n')
            else:
                theFile.write('Unet con elu y he_normal (solo FLAIR),' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('Unet con elu y he_normal (solo FLAIR),' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('Unet con elu y he_normal (solo FLAIR),' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('Unet con elu y he_normal (solo FLAIR),' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('Unet con elu y he_normal (solo FLAIR),' + str(key) + ',Lesion F1,' + str(f1) + '\n')
    
# comparison of models only using a FLAIR input
def comparison_onlyFlairTestData(lang="en"):
 
    filePath = "results-en-FLAIR.csv" if lang == "en" else "results-es-FLAIR.csv"
 
    with open(filePath, 'w') as theFile:
        if lang == "en":
            theFile.write('Applied model,patient,metric,value\n')
        else:
            theFile.write('Modelo aplicado,patient,metric,value\n') 
        theDict3 = test_leave_one_out_all(t1=False) 
        for key in theDict3:
            dsc = theDict3[key][0]
            h95 = theDict3[key][1]
            avd = theDict3[key][2]
            recall = theDict3[key][3]
            f1 = theDict3[key][4]
            print('Result of patient ' + str(key))
            print('Dice',                dsc,       '(higher is better, max=1)')
            print('HD',                  h95, 'mm',  '(lower is better, min=0)')
            print('AVD',                 avd,  '%',  '(lower is better, min=0)')
            print('Lesion detection', recall,       '(higher is better, max=1)')
            print('Lesion F1',            f1,       '(higher is better, max=1)')
            if lang == "en":
                theFile.write('State of the art (FLAIR only),' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('State of the art (FLAIR only),' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('State of the art (FLAIR only),' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('State of the art (FLAIR only),' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('State of the art (FLAIR only),' + str(key) + ',Lesion F1,' + str(f1) + '\n')
            else:
                theFile.write('Exp. solo MRI FLAIR,' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('Exp. solo MRI FLAIR,' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('Exp. solo MRI FLAIR,' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('Exp. solo MRI FLAIR,' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('Exp. solo MRI FLAIR,' + str(key) + ',Lesion F1,' + str(f1) + '\n')

    
        theDict4 = test_leave_one_out_all(t1=True) 
        for key in theDict4:
            dsc = theDict4[key][0]
            h95 = theDict4[key][1]
            avd = theDict4[key][2]
            recall = theDict4[key][3]
            f1 = theDict4[key][4]
            print('Result of patient ' + str(key))
            print('Dice',                dsc,       '(higher is better, max=1)')
            print('HD',                  h95, 'mm',  '(lower is better, min=0)')
            print('AVD',                 avd,  '%',  '(lower is better, min=0)')
            print('Lesion detection', recall,       '(higher is better, max=1)')
            print('Lesion F1',            f1,       '(higher is better, max=1)')
            if lang == "en":
                theFile.write('State of the art (Combined),' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('State of the art (Combined),' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('State of the art (Combined),' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('State of the art (Combined),' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('State of the art (Combined),' + str(key) + ',Lesion F1,' + str(f1) + '\n')
            else:
                theFile.write('Exp. solo MRI FLAIR ,' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('Exp. solo MRI FLAIR ,' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('Exp. solo MRI FLAIR ,' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('Exp. solo MRI FLAIR ,' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('Exp. solo MRI FLAIR ,' + str(key) + ',Lesion F1,' + str(f1) + '\n')

        theDict5 = test_leave_one_out_all_unet2(t1=False)
        for key in theDict5:
            dsc = theDict5[key][0]
            h95 = theDict5[key][1]
            avd = theDict5[key][2]
            recall = theDict5[key][3]
            f1 = theDict5[key][4]
            print('Result of patient ' + str(key))
            print('Dice',                dsc,       '(higher is better, max=1)')
            print('HD',                  h95, 'mm',  '(lower is better, min=0)')
            print('AVD',                 avd,  '%',  '(lower is better, min=0)')
            print('Lesion detection', recall,       '(higher is better, max=1)')
            print('Lesion F1',            f1,       '(higher is better, max=1)')
            if lang == "en":
                theFile.write('U-net #2 (FLAIR only),' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('U-net #2 (FLAIR only),' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('U-net #2 (FLAIR only),' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('U-net #2 (FLAIR only),' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('U-net #2 (FLAIR only),' + str(key) + ',Lesion F1,' + str(f1) + '\n')
            else:
                theFile.write('U-net #2 (solo FLAIR),' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('U-net #2 (solo FLAIR),' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('U-net #2 (solo FLAIR),' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('U-net #2 (solo FLAIR),' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('U-net #2 (solo FLAIR),' + str(key) + ',Lesion F1,' + str(f1) + '\n')

        theDict6 = test_leave_one_out_all_unet2(t1=True)
        for key in theDict:
            dsc = theDict6[key][0]
            h95 = theDict6[key][1]
            avd = theDict6[key][2]
            recall = theDict6[key][3]
            f1 = theDict6[key][4]
            print('Result of patient ' + str(key))
            print('Dice',                dsc,       '(higher is better, max=1)')
            print('HD',                  h95, 'mm',  '(lower is better, min=0)')
            print('AVD',                 avd,  '%',  '(lower is better, min=0)')
            print('Lesion detection', recall,       '(higher is better, max=1)')
            print('Lesion F1',            f1,       '(higher is better, max=1)')
            if lang == "en":
                theFile.write('U-net #2 (combined),' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('U-net #2 (combined),' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('U-net #2 (combined),' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('U-net #2 (combined),' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('U-net #2 (combined),' + str(key) + ',Lesion F1,' + str(f1) + '\n')
            else:
                theFile.write('U-net #2 (Combined),' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('U-net #2 (Combined),' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('U-net #2 (Combined),' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('U-net #2 (Combined),' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('U-net #2 (Combined),' + str(key) + ',Lesion F1,' + str(f1) + '\n')
    
        theDict7 = test_leave_one_out_all_unet3(t1=True)
        for key in theDict:
            dsc = theDict7[key][0]
            h95 = theDict7[key][1]
            avd = theDict7[key][2]
            recall = theDict7[key][3]
            f1 = theDict7[key][4]
            print('Result of patient ' + str(key))
            print('Dice',                dsc,       '(higher is better, max=1)')
            print('HD',                  h95, 'mm',  '(lower is better, min=0)')
            print('AVD',                 avd,  '%',  '(lower is better, min=0)')
            print('Lesion detection', recall,       '(higher is better, max=1)')
            print('Lesion F1',            f1,       '(higher is better, max=1)')
            if lang == "en":
                theFile.write('U-net #3 (FLAIR only),' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('U-net #3 (FLAIR only),' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('U-net #3 (FLAIR only),' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('U-net #3 (FLAIR only),' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('U-net #3 (FLAIR only),' + str(key) + ',Lesion F1,' + str(f1) + '\n')
            else:
                theFile.write('Unet con elu y he_normal (solo FLAIR),' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('Unet con elu y he_normal (solo FLAIR),' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('Unet con elu y he_normal (solo FLAIR),' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('Unet con elu y he_normal (solo FLAIR),' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('Unet con elu y he_normal (solo FLAIR),' + str(key) + ',Lesion F1,' + str(f1) + '\n')
    
        theDict8 = test_leave_one_out_all_unet3(t1=False)
        for key in theDict:
            dsc = theDict8[key][0]
            h95 = theDict8[key][1]
            avd = theDict8[key][2]
            recall = theDict8[key][3]
            f1 = theDict8[key][4]
            print('Result of patient ' + str(key))
            print('Dice',                dsc,       '(higher is better, max=1)')
            print('HD',                  h95, 'mm',  '(lower is better, min=0)')
            print('AVD',                 avd,  '%',  '(lower is better, min=0)')
            print('Lesion detection', recall,       '(higher is better, max=1)')
            print('Lesion F1',            f1,       '(higher is better, max=1)')
            if lang == "en":
                theFile.write('U-net #3 (combined),' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('U-net #3 (combined),' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('U-net #3 (combined),' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('U-net #3 (combined),' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('U-net #3 (combined),' + str(key) + ',Lesion F1,' + str(f1) + '\n')
            else:
                theFile.write('Unet con elu y he_normal (combined),' + str(key) + ',Dice,' + str(dsc) + '\n')
                theFile.write('Unet con elu y he_normal (combined),' + str(key) + ',HD,' + str(h95) + '\n')
                theFile.write('Unet con elu y he_normal (combined),' + str(key) + ',AVD,' + str(avd) + '\n')
                theFile.write('Unet con elu y he_normal (combined),' + str(key) + ',Lesion detection,' + str(recall) + '\n')
                theFile.write('Unet con elu y he_normal (combined),' + str(key) + ',Lesion F1,' + str(f1) + '\n')

def general_preprocessing(FLAIR_image, T1_image):
    channel_num = 2
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]
    T1_image = np.float32(T1_image)

    if (image_rows_Dataset >= rows_standard and image_cols_Dataset >= cols_standard):
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
        return imgs_two_channels
    elif (image_rows_Dataset >= rows_standard and image_cols_Dataset < cols_standard):

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
        FLAIR_image_suitable[:, :, (cols_standard-image_cols_Dataset)/2:(cols_standard+image_cols_Dataset)/2] = FLAIR_image[:, (image_rows_Dataset/2-rows_standard/2):(image_rows_Dataset/2+rows_standard/2), :]
    
        # T1 -----------------------------------------------
        brain_mask_T1[T1_image >=thresh_T1] = 1
        brain_mask_T1[T1_image < thresh_T1] = 0
        for iii in range(np.shape(T1_image)[0]):
    
            brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
            #------Gaussion Normalization
        T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      #Gaussion Normalization
        T1_image /=np.std(T1_image[brain_mask_T1 == 1])

        T1_image_suitable[...] = np.min(T1_image)
        T1_image_suitable[:, :, (cols_standard-image_cols_Dataset)/2:(cols_standard+image_cols_Dataset)/2] = T1_image[:, (image_rows_Dataset/2-rows_standard/2):(image_rows_Dataset/2+rows_standard/2), :]
        #---------------------------------------------------
        FLAIR_image_suitable  = FLAIR_image_suitable[..., np.newaxis]
        T1_image_suitable  = T1_image_suitable[..., np.newaxis]
        
        imgs_two_channels = np.concatenate((FLAIR_image_suitable, T1_image_suitable), axis = 3)
        return imgs_two_channels
    elif (image_rows_Dataset < rows_standard and image_cols_Dataset >= cols_standard):
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
        FLAIR_image_suitable[:, (rows_standard - image_rows_Dataset)/2:(rows_standard + image_rows_Dataset)/2,:] = FLAIR_image[:, :, (image_cols_Dataset/2-cols_standard/2):(image_cols_Dataset/2+cols_standard/2)]
    
        # T1 -----------------------------------------------
        brain_mask_T1[T1_image >=thresh_T1] = 1
        brain_mask_T1[T1_image < thresh_T1] = 0
        for iii in range(np.shape(T1_image)[0]):
    
            brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
            #------Gaussion Normalization
        T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      #Gaussion Normalization
        T1_image /=np.std(T1_image[brain_mask_T1 == 1])

        T1_image_suitable[...] = np.min(T1_image)
        T1_image_suitable[:,(rows_standard - image_rows_Dataset)/2:(rows_standard + image_rows_Dataset)/2,:] = T1_image[:, :, (image_cols_Dataset/2-cols_standard/2):(image_cols_Dataset/2+cols_standard/2)]
        #---------------------------------------------------
        FLAIR_image_suitable  = FLAIR_image_suitable[..., np.newaxis]
        T1_image_suitable  = T1_image_suitable[..., np.newaxis]
        
        imgs_two_channels = np.concatenate((FLAIR_image_suitable, T1_image_suitable), axis = 3)
        return imgs_two_channels
    else:
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
        FLAIR_image_suitable[:, (rows_standard - image_rows_Dataset)/2:(rows_standard + image_rows_Dataset)/2,(cols_standard-image_cols_Dataset)/2:(cols_standard+image_cols_Dataset)/2] = FLAIR_image[...]
    
        # T1 -----------------------------------------------
        brain_mask_T1[T1_image >=thresh_T1] = 1
        brain_mask_T1[T1_image < thresh_T1] = 0
        for iii in range(np.shape(T1_image)[0]):
    
            brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
            #------Gaussion Normalization
        T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      #Gaussion Normalization
        T1_image /=np.std(T1_image[brain_mask_T1 == 1])

        T1_image_suitable[...] = np.min(T1_image)
        T1_image_suitable[:, (rows_standard - image_rows_Dataset)/2:(rows_standard + image_rows_Dataset)/2,(cols_standard-image_cols_Dataset)/2:(cols_standard+image_cols_Dataset)/2] = T1_image[...]
        #---------------------------------------------------
        FLAIR_image_suitable  = FLAIR_image_suitable[..., np.newaxis]
        T1_image_suitable  = T1_image_suitable[..., np.newaxis]
        
        imgs_two_channels = np.concatenate((FLAIR_image_suitable, T1_image_suitable), axis = 3)
        return imgs_two_channels


def general_postprocessing(FLAIR_array, pred):
    start_slice = 6
    num_selected_slice = np.shape(FLAIR_array)[0]
    image_rows_Dataset = np.shape(FLAIR_array)[1]
    image_cols_Dataset = np.shape(FLAIR_array)[2]
    original_pred = np.ndarray(np.shape(FLAIR_array), dtype=np.float32)
    original_pred[...] = 0

    if (image_rows_Dataset >= rows_standard and image_cols_Dataset >= cols_standard):
        original_pred[:,(image_rows_Dataset-rows_standard)/2:(image_rows_Dataset+rows_standard)/2,(image_cols_Dataset-cols_standard)/2:(image_cols_Dataset+cols_standard)/2] = pred[:,:,:,0]
        
        original_pred[0:start_slice, :, :] = 0
        original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
        return original_pred
    elif (image_rows_Dataset >= rows_standard and image_cols_Dataset < cols_standard):
        original_pred[:, (image_rows_Dataset-rows_standard)/2:(image_rows_Dataset+rows_standard)/2,:] = pred[:,:, (cols_standard-image_cols_Dataset)/2:(cols_standard+image_cols_Dataset)/2,0]

        original_pred[0:start_slice, :, :] = 0
        original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
        return original_pred


    elif (image_rows_Dataset < rows_standard and image_cols_Dataset >= cols_standard):
        original_pred[:, :,(image_cols_Dataset-cols_standard)/2:(image_cols_Dataset+cols_standard)/2] = pred[:,(rows_standard-image_rows_Dataset)/2:(rows_standard+image_rows_Dataset)/2,:,0]

        original_pred[0:start_slice, :, :] = 0
        original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
        return original_pred

    else:
        original_pred = pred[:,(rows_standard-image_rows_Dataset)/2:(rows_standard+image_rows_Dataset)/2,(cols_standard-image_cols_Dataset)/2:(cols_standard+image_cols_Dataset)/2,0]

        original_pred[0:start_slice, :, :] = 0
        original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
        return original_pred


def comparison_preprocessing(flair=True, t1=True, first5=True, verbose=True):
    theImages = [5,8,21,28,30,37,47,57,58] # actual images will be 5, 8, 21, 28, 30, 37
    theDict = {}
    for patient in theImages:
        if patient < 20: dir = '../../Data/originalData/Utrecht/'
        elif patient < 40: dir = '../../Data/originalData/Singapore/'
        else: dir = '../../Data/originalData/GE3T/'
        dirs = os.listdir(dir)
        dirs.sort()
        dir += dirs[patient%20]
        FLAIR_image = sitk.ReadImage(dir + '/pre/FLAIR.nii.gz')
        T1_image = sitk.ReadImage(dir + '/pre/T1.nii.gz')
        FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
        T1_array = sitk.GetArrayFromImage(T1_image)
        if patient < 40: imgs_test = Utrecht_preprocessing(FLAIR_array, T1_array)
        else: imgs_test = GE3T_preprocessing(FLAIR_array, T1_array)
        if not flair: imgs_test = imgs_test[..., 1:2].copy()
        if not t1: imgs_test = imgs_test[..., 0:1].copy()
        img_shape = (rows_standard, cols_standard, flair+t1)
        model = get_unet_2(img_shape, first5)
        model_path = 'models/unet2/'
        #if you want to test three models ensemble, just do like this: pred = (pred_1+pred_2+pred_3)/3
        model.load_weights(model_path + '_85_15_unet2.h5')
        pred = model.predict(imgs_test, batch_size=1, verbose=verbose)
        pred[pred > 0.5] = 1.
        pred[pred <= 0.5] = 0.
        if patient < 40: original_pred = Utrecht_postprocessing(FLAIR_array, pred)
        else: original_pred = GE3T_postprocessing(FLAIR_array, pred)
        filename_resultImage = model_path + str(patient) + '.nii.gz'
        sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage )
        filename_testImage = os.path.join(dir + '/wmh.nii.gz')
        testImage, resultImage = getImages(filename_testImage, filename_resultImage)
        dsc = getDSC(testImage, resultImage)
        avd = getAVD(testImage, resultImage) 
        h95 = getHausdorff(testImage, resultImage)
        recall, f1 = getLesionDetection(testImage, resultImage)
        theDict[patient] = [dsc, h95, avd, recall, f1]
    theDict2 = {}
    for patient in theImages:
        if patient < 20: dir = '../../Data/originalData/Utrecht/'
        elif patient < 40: dir = '../../Data/originalData/Singapore/'
        else: dir = '../../Data/originalData/GE3T/'
        dirs = os.listdir(dir)
        dirs.sort()
        dir += dirs[patient%20]
        FLAIR_image = sitk.ReadImage(dir + '/pre/FLAIR.nii.gz')
        T1_image = sitk.ReadImage(dir + '/pre/T1.nii.gz')
        FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
        T1_array = sitk.GetArrayFromImage(T1_image)
        imgs_test = general_preprocessing(FLAIR_array, T1_array)
        if not flair: imgs_test = imgs_test[..., 1:2].copy()
        if not t1: imgs_test = imgs_test[..., 0:1].copy()
        img_shape = (rows_standard, cols_standard, flair+t1)
        model = get_unet_2(img_shape, first5)
        model_path = 'models/unet2/'
        #if you want to test three models ensemble, just do like this: pred = (pred_1+pred_2+pred_3)/3
        model.load_weights(model_path + '_85_15_unet2.h5')
        pred = model.predict(imgs_test, batch_size=1, verbose=verbose)
        pred[pred > 0.5] = 1.
        pred[pred <= 0.5] = 0.
        original_pred = general_postprocessing(FLAIR_array, pred)
        filename_resultImage = model_path + str(patient) + '.nii.gz'
        sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage)
        filename_testImage = os.path.join(dir + '/wmh.nii.gz')
        testImage, resultImage = getImages(filename_testImage, filename_resultImage)
        dsc = getDSC(testImage, resultImage)
        avd = getAVD(testImage, resultImage) 
        h95 = getHausdorff(testImage, resultImage)
        recall, f1 = getLesionDetection(testImage, resultImage)
        theDict2[patient] = [dsc, h95, avd, recall, f1]
    
    filePath = "results-preprocessed-unet2.csv"
    with open(filePath, 'w') as theFile:
        theFile.write('Applied model,patient,metric,value\n')


        for key in theDict:
            dsc = theDict[key][0]
            h95 = theDict[key][1]
            avd = theDict[key][2]
            recall = theDict[key][3]
            f1 = theDict[key][4]
            print('Result of patient ' + str(key))
            print('Dice',                dsc,       '(higher is better, max=1)')
            print('HD',                  h95, 'mm',  '(lower is better, min=0)')
            print('AVD',                 avd,  '%',  '(lower is better, min=0)')
            print('Lesion detection', recall,       '(higher is better, max=1)')
            print('Lesion F1',            f1,       '(higher is better, max=1)')
            theFile.write('Normal Preprocessing,' + str(key) + ',Dice,' + str(dsc) + '\n')
            theFile.write('Normal Preprocessing,' + str(key) + ',HD,' + str(h95) + '\n')
            theFile.write('Normal Preprocessing,' + str(key) + ',AVD,' + str(avd) + '\n')
            theFile.write('Normal Preprocessing,' + str(key) + ',Lesion detection,' + str(recall) + '\n')
            theFile.write('Normal Preprocessing,' + str(key) + ',Lesion F1,' + str(f1) + '\n')


        for key in theDict:
            dsc = theDict[key][0]
            h95 = theDict[key][1]
            avd = theDict[key][2]
            recall = theDict[key][3]
            f1 = theDict[key][4]
            print('Result of patient ' + str(key))
            print('Dice',                dsc,       '(higher is better, max=1)')
            print('HD',                  h95, 'mm',  '(lower is better, min=0)')
            print('AVD',                 avd,  '%',  '(lower is better, min=0)')
            print('Lesion detection', recall,       '(higher is better, max=1)')
            print('Lesion F1',            f1,       '(higher is better, max=1)')
            theFile.write('Normal Preprocessing,' + str(key) + ',Dice,' + str(dsc) + '\n')
            theFile.write('Normal Preprocessing,' + str(key) + ',HD,' + str(h95) + '\n')
            theFile.write('Normal Preprocessing,' + str(key) + ',AVD,' + str(avd) + '\n')
            theFile.write('Normal Preprocessing,' + str(key) + ',Lesion detection,' + str(recall) + '\n')
            theFile.write('Normal Preprocessing,' + str(key) + ',Lesion F1,' + str(f1) + '\n')

        for key in theDict2:
            dsc = theDict2[key][0]
            h95 = theDict2[key][1]
            avd = theDict2[key][2]
            recall = theDict2[key][3]
            f1 = theDict2[key][4]
            print('Result of patient ' + str(key))
            print('Dice',                dsc,       '(higher is better, max=1)')
            print('HD',                  h95, 'mm',  '(lower is better, min=0)')
            print('AVD',                 avd,  '%',  '(lower is better, min=0)')
            print('Lesion detection', recall,       '(higher is better, max=1)')
            print('Lesion F1',            f1,       '(higher is better, max=1)')
            theFile.write('My Preprocessing,' + str(key) + ',Dice,' + str(dsc) + '\n')
            theFile.write('My Preprocessing,' + str(key) + ',HD,' + str(h95) + '\n')
            theFile.write('My Preprocessing,' + str(key) + ',AVD,' + str(avd) + '\n')
            theFile.write('My Preprocessing,' + str(key) + ',Lesion detection,' + str(recall) + '\n')
            theFile.write('My Preprocessing,' + str(key) + ',Lesion F1,' + str(f1) + '\n')


    

if __name__=='__main__':
    #comparison(lang="es")
    comparison_preprocessing()