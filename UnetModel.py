from __future__ import print_function
from imutils import paths
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import keras
from datetime import datetime

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
from keras.initializers import Ones, Zeros

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

batch_norm = False
layer_norm = True

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


data_gen_args = dict(
#    samplewise_center = True,
#    samplewise_std_normalization = True,
    rotation_range=180,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    vertical_flip = True,
    fill_mode='nearest')

#Validation data generation
data_val_gen_args = dict(
    #samplewise_center = True,
    #samplewise_std_normalization = True
    )
      
        
class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape

    
def conv_block(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    if batch_norm:
        conv = BatchNormalization(name=prefix + "_bn")(conv)
    if layer_norm:
        conv = LayerNormalization()(conv)
    conv = LeakyReLU(alpha=0.001, name=prefix + "_activation")(conv)
    return conv

def double_block(prevlayer, filters, prefix, strides=(1, 1)):
    layer1 = conv_block(prevlayer, filters, prefix+"1", strides)
    layer2 = conv_block(layer1   , filters, prefix+"2", strides)
    return layer2

def up_sampling_block(up_sampling_layer, left_skip_layer, filters, prefix, strides = (1,1)):
    up_layer = concatenate([UpSampling2D(size=(2, 2))(up_sampling_layer), left_skip_layer], axis=3)
    double_block_layer = double_block(up_layer, filters, prefix, strides)
    return double_block_layer


#Define loss function
def jaccard_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return  (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def jaccard_coef_loss(y_true, y_pred):
    j = -jaccard_coef(y_true, y_pred)
    return j


def fl_loss(y_true, y_pred):
    epsilon = 0.00001
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    j = -K.sum(y_true_f * K.log(y_pred_f + epsilon) + (1 - y_true_f) * K.log(1 - y_pred_f + epsilon)) #/ K.int_shape(y_pred)[0]
    l = - np.power(1-j, 0.5) * K.log(j + epsilon)
    return j

def focal_loss_fixed(y_true, y_pred):
    alpha = 0.25
    gamma = 0.5
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

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

    model.compile(optimizer=Adam(lr=5e-5), loss=fl_loss, metrics=[jaccard_coef])

    return model
    
def BiggerLeakyUnetModelWithBatchnorm():
    inputs = Input((img_rows, img_cols,3))
    conv1 = Conv2D(32, (3, 3), padding="same")(inputs)
    norm1 = BatchNormalization()(conv1)
    acti1 = LeakyReLU(alpha=0.001)(norm1)
    conv1 = Conv2D(32, (3, 3), padding="same")(acti1)
    norm1 = BatchNormalization()(conv1)
    acti1 = LeakyReLU(alpha=0.001)(norm1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(acti1)

    conv2 = Conv2D(64, (3, 3), padding="same")(pool1)
    norm2 = BatchNormalization()(conv2)
    acti2 = LeakyReLU(alpha=0.001)(norm2)
    conv2 = Conv2D(64, (3, 3), padding="same")(acti2)
    norm2 = BatchNormalization()(conv2)
    acti2 = LeakyReLU(alpha=0.001)(norm2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(acti2)

    conv3 = Conv2D(128, (3, 3), padding="same")(pool2)
    norm3 = BatchNormalization()(conv3)
    acti3 = LeakyReLU(alpha=0.001)(norm3)
    conv3 = Conv2D(128, (3, 3), padding="same")(acti3)
    norm3 = BatchNormalization()(conv3)
    acti3 = LeakyReLU(alpha=0.001)(norm3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(acti3)

    conv4 = Conv2D(256, (3, 3), padding="same")(pool3)
    norm4 = BatchNormalization()(conv4)
    acti4 = LeakyReLU(alpha=0.001)(norm4)
    conv4 = Conv2D(256, (3, 3), padding="same")(acti4)
    norm4 = BatchNormalization()(conv4)
    acti4 = LeakyReLU(alpha=0.001)(norm4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(acti4)

    conv5 = Conv2D(512, (3, 3), padding="same")(pool4)
    norm5 = BatchNormalization()(conv5)
    acti5 = LeakyReLU(alpha=0.001)(norm5)
    conv5 = Conv2D(512, (3, 3), padding="same")(acti5)
    norm5 = BatchNormalization()(conv5)
    acti5 = LeakyReLU(alpha=0.001)(norm5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(acti5)

    conv6 = Conv2D(1024, (3, 3), padding="same")(pool5)
    norm6 = BatchNormalization()(conv6)
    acti6 = LeakyReLU(alpha=0.001)(norm6)
    conv6 = Conv2D(1024, (3, 3), padding="same")(acti6)
    norm6 = BatchNormalization()(conv6)
    acti6 = LeakyReLU(alpha=0.001)(norm6)
    pool6 = MaxPooling2D(pool_size=(2, 2))(acti6)
    
    conv7 = Conv2D(2048, (3, 3), padding="same")(pool6)
    norm7 = BatchNormalization()(conv7)
    acti7 = LeakyReLU(alpha=0.001)(norm7)
    conv7 = Conv2D(2048, (3, 3), padding="same")(acti7)
    norm7 = BatchNormalization()(conv7)
    acti7 = LeakyReLU(alpha=0.001)(norm7)

    right_up6 = concatenate([UpSampling2D(size=(2, 2))(acti7), acti6], axis=3)
    right_conv6 = Conv2D(512, (3, 3), padding="same")(right_up6)
    right_norm6 = BatchNormalization()(right_conv6)
    right_acti6 = LeakyReLU(alpha=0.001)(right_norm6)
    right_conv6 = Conv2D(512, (3, 3), padding="same")(right_acti6)
    right_norm6 = BatchNormalization()(right_conv6)
    right_acti6 = LeakyReLU(alpha=0.001)(right_norm6)

    right_up5 = concatenate([UpSampling2D(size=(2, 2))(right_acti6), acti5], axis=3)
    right_conv5 = Conv2D(512, (3, 3), padding="same")(right_up5)
    right_norm5 = BatchNormalization()(right_conv5)
    right_acti5 = LeakyReLU(alpha=0.001)(right_norm5)
    right_conv5 = Conv2D(512, (3, 3), padding="same")(right_acti5)
    right_norm5 = BatchNormalization()(right_conv5)
    right_acti5 = LeakyReLU(alpha=0.001)(right_norm5)

    right_up4 = concatenate([UpSampling2D(size=(2, 2))(right_acti5), acti4], axis=3)
    right_conv4 = Conv2D(256, (3, 3), padding="same")(right_up4)
    right_norm4 = BatchNormalization()(right_conv4)
    right_acti4 = LeakyReLU(alpha=0.001)(right_norm4)
    right_conv4 = Conv2D(256, (3, 3), padding="same")(right_acti4)
    right_norm4 = BatchNormalization()(right_conv4)
    right_acti4 = LeakyReLU(alpha=0.001)(right_norm4)

    right_up3 = concatenate([UpSampling2D(size=(2, 2))(right_acti4), acti3], axis=3)
    right_conv3 = Conv2D(128, (3, 3), padding="same")(right_up3)
    right_norm3 = BatchNormalization()(right_conv3)
    right_acti3 = LeakyReLU(alpha=0.001)(right_norm3)
    right_conv3 = Conv2D(128, (3, 3), padding="same")(right_acti3)
    right_norm3 = BatchNormalization()(right_conv3)
    right_acti3 = LeakyReLU(alpha=0.001)(right_norm3)

    right_up2 = concatenate([UpSampling2D(size=(2, 2))(right_acti3), acti2], axis=3)
    right_conv2 = Conv2D(64, (3, 3), padding="same")(right_up2)
    right_norm2 = BatchNormalization()(right_conv2)
    right_acti2 = LeakyReLU(alpha=0.001)(right_norm2)
    right_conv2 = Conv2D(64, (3, 3), padding="same")(right_acti2)
    right_norm2 = BatchNormalization()(right_conv2)
    right_acti2 = LeakyReLU(alpha=0.001)(right_norm2)

    right_up1 = concatenate([UpSampling2D(size=(2, 2))(right_acti2), acti1], axis=3)
    right_conv1 = Conv2D(32, (3, 3), padding="same")(right_up1)
    right_norm1 = BatchNormalization()(right_conv1)
    right_acti1 = LeakyReLU(alpha=0.001)(right_norm1)
    right_conv1 = Conv2D(32, (3, 3), padding="same")(right_acti1)
    right_norm1 = BatchNormalization()(right_conv1)
    right_acti1 = LeakyReLU(alpha=0.001)(right_norm1)

    output1 = Conv2D(1, (1, 1))(right_acti1)
    output_norm = BatchNormalization()(output1)
    output = Activation("sigmoid")(output_norm)

    model = Model(input=inputs, output=output)

    model.compile(optimizer=Adam(lr=1e-4), loss=jaccard_coef_loss, metrics=[jaccard_coef])

    return model

def BiggerLeakyUnetModelWithLayernorm():
    inputs = Input((img_rows, img_cols,3))
    conv1 = Conv2D(32, (3, 3), padding="same")(inputs)
    norm1 = LayerNormalization()(conv1)
    acti1 = LeakyReLU(alpha=0.001)(norm1)
    conv1 = Conv2D(32, (3, 3), padding="same")(acti1)
    norm1 = LayerNormalization()(conv1)
    acti1 = LeakyReLU(alpha=0.001)(norm1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(acti1)

    conv2 = Conv2D(64, (3, 3), padding="same")(pool1)
    norm2 = LayerNormalization()(conv2)
    acti2 = LeakyReLU(alpha=0.001)(norm2)
    conv2 = Conv2D(64, (3, 3), padding="same")(acti2)
    norm2 = LayerNormalization()(conv2)
    acti2 = LeakyReLU(alpha=0.001)(norm2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(acti2)

    conv3 = Conv2D(128, (3, 3), padding="same")(pool2)
    norm3 = LayerNormalization()(conv3)
    acti3 = LeakyReLU(alpha=0.001)(norm3)
    conv3 = Conv2D(128, (3, 3), padding="same")(acti3)
    norm3 = LayerNormalization()(conv3)
    acti3 = LeakyReLU(alpha=0.001)(norm3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(acti3)

    conv4 = Conv2D(256, (3, 3), padding="same")(pool3)
    norm4 = LayerNormalization()(conv4)
    acti4 = LeakyReLU(alpha=0.001)(norm4)
    conv4 = Conv2D(256, (3, 3), padding="same")(acti4)
    norm4 = LayerNormalization()(conv4)
    acti4 = LeakyReLU(alpha=0.001)(norm4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(acti4)

    conv5 = Conv2D(512, (3, 3), padding="same")(pool4)
    norm5 = LayerNormalization()(conv5)
    acti5 = LeakyReLU(alpha=0.001)(norm5)
    conv5 = Conv2D(512, (3, 3), padding="same")(acti5)
    norm5 = LayerNormalization()(conv5)
    acti5 = LeakyReLU(alpha=0.001)(norm5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(acti5)

    conv6 = Conv2D(1024, (3, 3), padding="same")(pool5)
    norm6 = LayerNormalization()(conv6)
    acti6 = LeakyReLU(alpha=0.001)(norm6)
    conv6 = Conv2D(1024, (3, 3), padding="same")(acti6)
    norm6 = LayerNormalization()(conv6)
    acti6 = LeakyReLU(alpha=0.001)(norm6)
    pool6 = MaxPooling2D(pool_size=(2, 2))(acti6)
    
    conv7 = Conv2D(2048, (3, 3), padding="same")(pool6)
    norm7 = LayerNormalization()(conv7)
    acti7 = LeakyReLU(alpha=0.001)(norm7)
    conv7 = Conv2D(2048, (3, 3), padding="same")(acti7)
    norm7 = LayerNormalization()(conv7)
    acti7 = LeakyReLU(alpha=0.001)(norm7)

    right_up6 = concatenate([UpSampling2D(size=(2, 2))(acti7), acti6], axis=3)
    right_conv6 = Conv2D(512, (3, 3), padding="same")(right_up6)
    right_norm6 = LayerNormalization()(right_conv6)
    right_acti6 = LeakyReLU(alpha=0.001)(right_norm6)
    right_conv6 = Conv2D(512, (3, 3), padding="same")(right_acti6)
    right_norm6 = LayerNormalization()(right_conv6)
    right_acti6 = LeakyReLU(alpha=0.001)(right_norm6)

    right_up5 = concatenate([UpSampling2D(size=(2, 2))(right_acti6), acti5], axis=3)
    right_conv5 = Conv2D(512, (3, 3), padding="same")(right_up5)
    right_norm5 = LayerNormalization()(right_conv5)
    right_acti5 = LeakyReLU(alpha=0.001)(right_norm5)
    right_conv5 = Conv2D(512, (3, 3), padding="same")(right_acti5)
    right_norm5 = LayerNormalization()(right_conv5)
    right_acti5 = LeakyReLU(alpha=0.001)(right_norm5)

    right_up4 = concatenate([UpSampling2D(size=(2, 2))(right_acti5), acti4], axis=3)
    right_conv4 = Conv2D(256, (3, 3), padding="same")(right_up4)
    right_norm4 = LayerNormalization()(right_conv4)
    right_acti4 = LeakyReLU(alpha=0.001)(right_norm4)
    right_conv4 = Conv2D(256, (3, 3), padding="same")(right_acti4)
    right_norm4 = LayerNormalization()(right_conv4)
    right_acti4 = LeakyReLU(alpha=0.001)(right_norm4)

    right_up3 = concatenate([UpSampling2D(size=(2, 2))(right_acti4), acti3], axis=3)
    right_conv3 = Conv2D(128, (3, 3), padding="same")(right_up3)
    right_norm3 = LayerNormalization()(right_conv3)
    right_acti3 = LeakyReLU(alpha=0.001)(right_norm3)
    right_conv3 = Conv2D(128, (3, 3), padding="same")(right_acti3)
    right_norm3 = LayerNormalization()(right_conv3)
    right_acti3 = LeakyReLU(alpha=0.001)(right_norm3)

    right_up2 = concatenate([UpSampling2D(size=(2, 2))(right_acti3), acti2], axis=3)
    right_conv2 = Conv2D(64, (3, 3), padding="same")(right_up2)
    right_norm2 = LayerNormalization()(right_conv2)
    right_acti2 = LeakyReLU(alpha=0.001)(right_norm2)
    right_conv2 = Conv2D(64, (3, 3), padding="same")(right_acti2)
    right_norm2 = LayerNormalization()(right_conv2)
    right_acti2 = LeakyReLU(alpha=0.001)(right_norm2)

    right_up1 = concatenate([UpSampling2D(size=(2, 2))(right_acti2), acti1], axis=3)
    right_conv1 = Conv2D(32, (3, 3), padding="same")(right_up1)
    right_norm1 = LayerNormalization()(right_conv1)
    right_acti1 = LeakyReLU(alpha=0.001)(right_norm1)
    right_conv1 = Conv2D(32, (3, 3), padding="same")(right_acti1)
    right_norm1 = LayerNormalization()(right_conv1)
    right_acti1 = LeakyReLU(alpha=0.001)(right_norm1)

    output1 = Conv2D(1, (1, 1))(right_acti1)
    output_norm = LayerNormalization()(output1)
    output = Activation("sigmoid")(output_norm)

    model = Model(input=inputs, output=output)

    model.compile(optimizer=Adam(lr=5e-5), loss=jaccard_coef_loss, metrics=[jaccard_coef])

    return model

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (224,224),seed = 1):
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
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (224,224),seed = 1):
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

        
def plotTrainigGraph(hist):
    # Plot training & validation accuracy values
    plt.plot(hist['jaccard_coef'])
    plt.plot(hist['val_jaccard_coef'])
    plt.title('Coefficiency')
    plt.ylabel('Coefficiency')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.legend(['Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.legend(['Validation'], loc='upper left')
    plt.show()

    coef = np.array(hist['jaccard_coef'])
    val_coef = np.array(hist['val_jaccard_coef'])
    print("Training co-effiency    : {};\nValidation co-effiency : {}".format(coef[coef==max(coef)][0], val_coef[np.argmax(coef)]))

def predictTestSet(model_location):
    model.load_weights(model_location)

    file_names = next(os.walk(test_data_dir))[2]
    scores = []
    for file in file_names:
        grey_img = load_img(os.path.join(test_data_dir,file), target_size=(img_rows, img_cols), grayscale=False)
        mask_img = load_img(os.path.join(test_data_mask_dir,file.split('.')[0]+"_segmentation.png"), 
                            target_size=(img_rows, img_cols), grayscale=True)
        img = img_to_array(grey_img)
        img_mask = img_to_array(mask_img)

        #Preprocess image mask
        #img_mask = img_mask /255
        #img_mask[img_mask > 0.5] = 1
        #img_mask[img_mask <= 0.5] = 0
        #Preprocess images
        #mean = np.mean(img)  # mean for data centering
        #std = np.std(img)  # std for data normalization
        #img -= mean
        #img /= std
        img, img_mask = normalizeData(img, img_mask)
        img = np.reshape(img,(1,)+img.shape)

        pred = model.predict([img])
        sess = tf.Session()
        score = sess.run(jaccard_coef(img_mask, pred))
        print("{} -- jaccard index: {}".format(file,score))
        scores.append([file,score])

        result_img = array_to_img(pred[0] * 255 )
        result_img.save(os.path.join(test_data_pred_dir, file.split('.')[0] + '_predict.jpg'))

    with open("unet_test_result.csv", 'w') as f:
        f.write("filename, jaccard_index\n")
        for i in range(len(scores)):
        #print(scores[i])
            f.write("{},{}\n".format(scores[i][0], scores[i][1]))

def showPredictResult(file, model):
    #file = 'data/train/images/ISIC_0000000.jpg'
    grey_img = load_img(os.path.join(test_data_dir,file), target_size=(img_rows, img_cols), grayscale=False)
    mask_img = load_img(os.path.join(test_data_mask_dir,file.split('.')[0]+"_segmentation.png"), 
                        target_size=(img_rows, img_cols), grayscale=True)
    #grey_img = load_img(file, target_size=(img_rows, img_cols), grayscale=False)
    #mask_img = load_img('data/train/masks/ISIC_0000000_segmentation.png', target_size=(img_rows, img_cols), grayscale=True)

    img = img_to_array(grey_img)
    img_mask = img_to_array(mask_img)

    img, img_mask = normalizeData(img, img_mask)
    img = np.reshape(img,(1,)+img.shape)

    pred = model.predict([img])
    sess = tf.Session()
    score = sess.run(jaccard_coef(img_mask, pred))
    print("{} -- jaccard index: {}".format(file,score))

    result_img = array_to_img(pred[0] * 255 )

    f, ax = plt.subplots(1,2, figsize = (50,50))
    ax[0].imshow(grey_img) 
    ax[0].axis('off')
    ax[0].set_title('image')
    ax[1].imshow(result_img)
    ax[1].axis('off')
    ax[1].set_title('mask')
    plt.show()
    

#Create folder for models
date_object = datetime.now()
# convert object to the format we want
formatted_date = date_object.strftime('%Y%m%d')
output_dir = 'unet/{}'.format(formatted_date)
os.makedirs(output_dir, exist_ok =True)

#Setup Checkpoint to only capture best estimate
model_checkpoint = ModelCheckpoint('{}/unet_lesion_{}_{{epoch:03d}}-{{val_jaccard_coef:.5f}}.hdf5'.format(output_dir, formatted_date)
                                   , monitor='val_jaccard_coef'
                                   ,verbose=1, mode='max', save_best_only=True)