from __future__ import print_function
from imutils import paths
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import keras

from keras.preprocessing import image as image_utils
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import backend as keras

root_dir = './data'
training_data_dir = os.path.join(root_dir, 'train/images')
training_data_mask_dir = os.path.join(root_dir, 'train/masks')

val_data_dir = os.path.join(root_dir, 'val/images')
val_data_pred_dir = os.path.join(root_dir, 'val/predict')
val_data_mask_dir = os.path.join(root_dir, 'val/masks')

test_data_dir = os.path.join(root_dir, 'test/images')
test_data_pred_dir = os.path.join(root_dir, 'test/predict')
test_data_mask_dir = os.path.join(root_dir, 'test/masks')

img_rows = 256
img_cols = 256

batch_size = 6

#Enable tensorboard
tensorBoard = TensorBoard(
    log_dir='./logs', 
    histogram_freq=0, 
    batch_size=batch_size, 
    write_graph=False, 
    write_grads=False, 
    write_images=False, 
    embeddings_freq=0)


#Define loss function
def jaccard_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return  (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def jaccard_coef_loss(y_true, y_pred):
    j = jaccard_coef(y_true, y_pred)
    l = - np.power(1-j, 3) * K.log(j)
    return l


def normalizeData(img,mask):
    mean = np.mean(img)  # mean for data centering
    std = np.std(img)  # std for data normalization
    img -= mean
    img /= std
    mask = mask /255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img,mask)
def normalizeData_rgb(img,mask):
    for i in range(3):
        mean = np.mean(img[:,:,i])  # mean for data centering
        std = np.std(img[:,:,i])  # std for data normalization
        img[:,:,i] -= mean
        img[:,:,i] /= std
    mask = mask /255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img,mask)

def UnetModel():
    inputs = Input((img_rows, img_cols,1))
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=jaccard_coef_loss, metrics=[jaccard_coef])

    return model
    
def FullUnetModel():
    inputs = Input((img_rows, img_cols,1))
    conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss=jaccard_coef_loss, metrics=[jaccard_coef])

    return model

def LeakyUnetModel():
    inputs = Input((img_rows, img_cols,1))
    conv1 = Conv2D(32, (3, 3), padding="same")(inputs)
    acti1 = LeakyReLU(alpha=0.001)(conv1)
    conv1 = Conv2D(32, (3, 3), padding="same")(acti1)
    acti1 = LeakyReLU(alpha=0.001)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(acti1)

    conv2 = Conv2D(64, (3, 3), padding="same")(pool1)
    acti2 = LeakyReLU(alpha=0.001)(conv2)
    conv2 = Conv2D(64, (3, 3), padding="same")(acti2)
    acti2 = LeakyReLU(alpha=0.001)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(acti2)

    conv3 = Conv2D(128, (3, 3), padding="same")(pool2)
    acti3 = LeakyReLU(alpha=0.001)(conv3)
    conv3 = Conv2D(128, (3, 3), padding="same")(acti3)
    acti3 = LeakyReLU(alpha=0.001)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(acti3)

    conv4 = Conv2D(256, (3, 3), padding="same")(pool3)
    acti4 = LeakyReLU(alpha=0.001)(conv4)
    conv4 = Conv2D(256, (3, 3), padding="same")(acti4)
    acti4 = LeakyReLU(alpha=0.001)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(acti4)

    conv5 = Conv2D(512, (3, 3), padding="same")(pool4)
    acti5 = LeakyReLU(alpha=0.001)(conv5)
    conv5 = Conv2D(512, (3, 3), padding="same")(acti5)
    acti5 = LeakyReLU(alpha=0.001)(conv5)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(acti5), acti4], axis=3)
    conv6 = Conv2D(256, (3, 3), padding="same")(up6)
    acti6 = LeakyReLU(alpha=0.001)(conv6)
    conv6 = Conv2D(256, (3, 3), padding="same")(acti6)
    acti6 = LeakyReLU(alpha=0.001)(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(acti6), acti3], axis=3)
    conv7 = Conv2D(128, (3, 3), padding="same")(up7)
    acti7 = LeakyReLU(alpha=0.001)(conv7)
    conv7 = Conv2D(128, (3, 3), padding="same")(acti7)
    acti7 = LeakyReLU(alpha=0.001)(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(acti7), acti2], axis=3)
    conv8 = Conv2D(64, (3, 3), padding="same")(up8)
    acti8 = LeakyReLU(alpha=0.001)(conv8)
    conv8 = Conv2D(64, (3, 3), padding="same")(acti8)
    acti8 = LeakyReLU(alpha=0.001)(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(acti8), acti1], axis=3)
    conv9 = Conv2D(32, (3, 3), padding="same")(up9)
    acti9 = LeakyReLU(alpha=0.001)(conv9)
    conv9 = Conv2D(32, (3, 3), padding="same")(acti9)
    acti9 = LeakyReLU(alpha=0.001)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(acti9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=3e-5), loss=jaccard_coef_loss, metrics=[jaccard_coef])

    return model
    
    
def BigLeakyUnetModel():
    inputs = Input((img_rows, img_cols,3))
    conv1 = Conv2D(32, (3, 3), padding="same")(inputs)
    acti1 = LeakyReLU(alpha=0.001)(conv1)
    conv1 = Conv2D(32, (3, 3), padding="same")(acti1)
    acti1 = LeakyReLU(alpha=0.001)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(acti1)

    conv2 = Conv2D(64, (3, 3), padding="same")(pool1)
    acti2 = LeakyReLU(alpha=0.001)(conv2)
    conv2 = Conv2D(64, (3, 3), padding="same")(acti2)
    acti2 = LeakyReLU(alpha=0.001)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(acti2)

    conv3 = Conv2D(128, (3, 3), padding="same")(pool2)
    acti3 = LeakyReLU(alpha=0.001)(conv3)
    conv3 = Conv2D(128, (3, 3), padding="same")(acti3)
    acti3 = LeakyReLU(alpha=0.001)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(acti3)

    conv4 = Conv2D(256, (3, 3), padding="same")(pool3)
    acti4 = LeakyReLU(alpha=0.001)(conv4)
    conv4 = Conv2D(256, (3, 3), padding="same")(acti4)
    acti4 = LeakyReLU(alpha=0.001)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(acti4)

    conv5 = Conv2D(512, (3, 3), padding="same")(pool4)
    acti5 = LeakyReLU(alpha=0.001)(conv5)
    conv5 = Conv2D(512, (3, 3), padding="same")(acti5)
    acti5 = LeakyReLU(alpha=0.001)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(acti5)

    conv6 = Conv2D(1024, (3, 3), padding="same")(pool5)
    acti6 = LeakyReLU(alpha=0.001)(conv6)
    conv6 = Conv2D(1024, (3, 3), padding="same")(acti6)
    acti6 = LeakyReLU(alpha=0.001)(conv6)
    
    right_up1 = concatenate([UpSampling2D(size=(2, 2))(acti6), acti5], axis=3)
    right_conv1 = Conv2D(512, (3, 3), padding="same")(right_up1)
    right_acti1 = LeakyReLU(alpha=0.001)(right_conv1)
    right_conv1 = Conv2D(512, (3, 3), padding="same")(right_acti1)
    right_acti1 = LeakyReLU(alpha=0.001)(right_conv1)

    right_up2 = concatenate([UpSampling2D(size=(2, 2))(right_acti1), acti4], axis=3)
    right_conv2 = Conv2D(256, (3, 3), padding="same")(right_up2)
    right_acti2 = LeakyReLU(alpha=0.001)(right_conv2)
    right_conv2 = Conv2D(256, (3, 3), padding="same")(right_acti2)
    right_acti2 = LeakyReLU(alpha=0.001)(right_conv2)

    right_up3 = concatenate([UpSampling2D(size=(2, 2))(right_acti2), acti3], axis=3)
    right_conv3 = Conv2D(128, (3, 3), padding="same")(right_up3)
    right_acti3 = LeakyReLU(alpha=0.001)(right_conv3)
    right_conv3 = Conv2D(128, (3, 3), padding="same")(right_acti3)
    right_acti3 = LeakyReLU(alpha=0.001)(right_conv3)

    right_up4 = concatenate([UpSampling2D(size=(2, 2))(right_acti3), acti2], axis=3)
    right_conv4 = Conv2D(64, (3, 3), padding="same")(right_up4)
    right_acti4 = LeakyReLU(alpha=0.001)(right_conv4)
    right_conv4 = Conv2D(64, (3, 3), padding="same")(right_acti4)
    right_acti4 = LeakyReLU(alpha=0.001)(right_conv4)

    right_up5 = concatenate([UpSampling2D(size=(2, 2))(right_acti4), acti1], axis=3)
    right_conv5 = Conv2D(32, (3, 3), padding="same")(right_up5)
    right_acti5 = LeakyReLU(alpha=0.001)(right_conv5)
    right_conv5 = Conv2D(32, (3, 3), padding="same")(right_acti5)
    right_acti5 = LeakyReLU(alpha=0.001)(right_conv5)

    output = Conv2D(1, (1, 1), activation='sigmoid')(right_acti5)

    model = Model(input=inputs, output=output)

    model.compile(optimizer=Adam(lr=5e-5), loss=jaccard_coef_loss, metrics=[jaccard_coef])

    return model
    
def BiggerLeakyUnetModel():
    inputs = Input((img_rows, img_cols,3))
    conv1 = Conv2D(32, (3, 3), padding="same")(inputs)
    acti1 = LeakyReLU(alpha=0.001)(conv1)
    conv1 = Conv2D(32, (3, 3), padding="same")(acti1)
    acti1 = LeakyReLU(alpha=0.001)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(acti1)

    conv2 = Conv2D(64, (3, 3), padding="same")(pool1)
    acti2 = LeakyReLU(alpha=0.001)(conv2)
    conv2 = Conv2D(64, (3, 3), padding="same")(acti2)
    acti2 = LeakyReLU(alpha=0.001)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(acti2)

    conv3 = Conv2D(128, (3, 3), padding="same")(pool2)
    acti3 = LeakyReLU(alpha=0.001)(conv3)
    conv3 = Conv2D(128, (3, 3), padding="same")(acti3)
    acti3 = LeakyReLU(alpha=0.001)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(acti3)

    conv4 = Conv2D(256, (3, 3), padding="same")(pool3)
    acti4 = LeakyReLU(alpha=0.001)(conv4)
    conv4 = Conv2D(256, (3, 3), padding="same")(acti4)
    acti4 = LeakyReLU(alpha=0.001)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(acti4)

    conv5 = Conv2D(512, (3, 3), padding="same")(pool4)
    acti5 = LeakyReLU(alpha=0.001)(conv5)
    conv5 = Conv2D(512, (3, 3), padding="same")(acti5)
    acti5 = LeakyReLU(alpha=0.001)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(acti5)

    conv6 = Conv2D(1024, (3, 3), padding="same")(pool5)
    acti6 = LeakyReLU(alpha=0.001)(conv6)
    conv6 = Conv2D(1024, (3, 3), padding="same")(acti6)
    acti6 = LeakyReLU(alpha=0.001)(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2))(acti6)
    
    conv7 = Conv2D(2048, (3, 3), padding="same")(pool6)
    acti7 = LeakyReLU(alpha=0.001)(conv7)
    conv7 = Conv2D(2048, (3, 3), padding="same")(acti7)
    acti7 = LeakyReLU(alpha=0.001)(conv7)

    right_up6 = concatenate([UpSampling2D(size=(2, 2))(acti7), acti6], axis=3)
    right_conv6 = Conv2D(512, (3, 3), padding="same")(right_up6)
    right_acti6 = LeakyReLU(alpha=0.001)(right_conv6)
    right_conv6 = Conv2D(512, (3, 3), padding="same")(right_acti6)
    right_acti6 = LeakyReLU(alpha=0.001)(right_conv6)

    right_up5 = concatenate([UpSampling2D(size=(2, 2))(right_acti6), acti5], axis=3)
    right_conv5 = Conv2D(512, (3, 3), padding="same")(right_up5)
    right_acti5 = LeakyReLU(alpha=0.001)(right_conv5)
    right_conv5 = Conv2D(512, (3, 3), padding="same")(right_acti5)
    right_acti5 = LeakyReLU(alpha=0.001)(right_conv5)

    right_up4 = concatenate([UpSampling2D(size=(2, 2))(right_acti5), acti4], axis=3)
    right_conv4 = Conv2D(256, (3, 3), padding="same")(right_up4)
    right_acti4 = LeakyReLU(alpha=0.001)(right_conv4)
    right_conv4 = Conv2D(256, (3, 3), padding="same")(right_acti4)
    right_acti4 = LeakyReLU(alpha=0.001)(right_conv4)

    right_up3 = concatenate([UpSampling2D(size=(2, 2))(right_acti4), acti3], axis=3)
    right_conv3 = Conv2D(128, (3, 3), padding="same")(right_up3)
    right_acti3 = LeakyReLU(alpha=0.001)(right_conv3)
    right_conv3 = Conv2D(128, (3, 3), padding="same")(right_acti3)
    right_acti3 = LeakyReLU(alpha=0.001)(right_conv3)

    right_up2 = concatenate([UpSampling2D(size=(2, 2))(right_acti3), acti2], axis=3)
    right_conv2 = Conv2D(64, (3, 3), padding="same")(right_up2)
    right_acti2 = LeakyReLU(alpha=0.001)(right_conv2)
    right_conv2 = Conv2D(64, (3, 3), padding="same")(right_acti2)
    right_acti2 = LeakyReLU(alpha=0.001)(right_conv2)

    right_up1 = concatenate([UpSampling2D(size=(2, 2))(right_acti2), acti1], axis=3)
    right_conv1 = Conv2D(32, (3, 3), padding="same")(right_up1)
    right_acti1 = LeakyReLU(alpha=0.001)(right_conv1)
    right_conv1 = Conv2D(32, (3, 3), padding="same")(right_acti1)
    right_acti1 = LeakyReLU(alpha=0.001)(right_conv1)

    output = Conv2D(1, (1, 1), activation='sigmoid')(right_acti1)

    model = Model(input=inputs, output=output)

    model.compile(optimizer=Adam(lr=5e-5), loss=jaccard_coef_loss, metrics=[jaccard_coef])

    return model

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = normalizeData(img,mask)
        yield (img,mask)

        
def validationGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = normalizeData(img,mask)
        yield (img,mask)
