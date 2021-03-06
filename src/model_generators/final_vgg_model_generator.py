#!/usr/bin/env python

from __future__ import print_function

import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import utils.deepmirna_utils as deep_utils

BATCH_SIZE = 128
NUM_CLASSES = 2
IMG_ROWS, IMG_COLUMNS = 25, 100

#32, 64, 128, 256
DENSE_UNITS = 256

def build_model_one_module_3x3():
    input_shape_img = deep_utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)

    block1_units = 48
    kernel_shape = (3, 3)
        
    model = Sequential()
    model.add(Conv2D(block1_units, kernel_size=kernel_shape,
                     activation='relu', input_shape=input_shape_img,
                     name='input_layer'))
    model.add(Conv2D(block1_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(DENSE_UNITS, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax',
                    name='logits_layer'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "vgg_one_module_3x3_dense{}".format(DENSE_UNITS)
    return (model, model_name)


def build_model_two_modules_3x3():
    input_shape_img = deep_utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)

    block1_units = 48
    block2_units = 60
    kernel_shape = (3, 3)
    
    model = Sequential()
    model.add(Conv2D(block1_units, kernel_size=kernel_shape,
                     activation='relu', input_shape=input_shape_img,
                     name='input_layer'))
    model.add(Conv2D(block1_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(block2_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(block2_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(DENSE_UNITS, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax',
                    name='logits_layer'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "vgg_two_modules_3x3_dense{}".format(DENSE_UNITS)
    return (model, model_name)

def build_model_three_modules_3x3():
    input_shape_img = deep_utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)

    block1_units = 48
    block2_units = 60
    block3_units = 72
    kernel_shape = (5, 5)
    
    model = Sequential()
    model.add(Conv2D(block1_units, kernel_size=kernel_shape,
                     activation='relu', input_shape=input_shape_img,
                     name='input_layer'))
    model.add(Conv2D(block1_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(block2_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(block2_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(block3_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(block3_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(DENSE_UNITS, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax',
                    name='logits_layer'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "vgg_three_modules_3x3_dense{}".format(DENSE_UNITS)
    return (model, model_name)


def build_model_four_modules_3x3():
    input_shape_img = deep_utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)

    block1_units = 48
    block2_units = 60
    block3_units = 72
    block4_units = 84
    kernel_shape = (3, 3)
    
    model = Sequential()
    model.add(Conv2D(block1_units, kernel_size=kernel_shape,
                     activation='relu', input_shape=input_shape_img,
                     name='input_layer'))
    model.add(Conv2D(block1_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(block2_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(block2_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(block3_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(block3_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(block3_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(block3_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(DENSE_UNITS, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax',
                    name='logits_layer'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "vgg_four_modules_3x3_dense{}".format(DENSE_UNITS)
    return (model, model_name)


## vgg_3m_256fcu
def build_vgg_3M_256FCU():
    input_shape_img = deep_utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)

    block1_units = 48
    block2_units = 60
    block3_units = 72
    kernel_shape = (3, 3)
    
    model = Sequential()
    model.add(Conv2D(block1_units, kernel_size=kernel_shape,
                     activation='relu', input_shape=input_shape_img,
                     name='input_layer'))
    model.add(Conv2D(block1_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(block2_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(block2_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(block3_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(block3_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax',
                    name='logits_layer'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "vgg_3M_256FCU"
    return (model, model_name)


## vgg_2m_256fcu
def build_vgg_2M_256FCU():
    input_shape_img = deep_utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)

    block1_units = 48
    block2_units = 60
    kernel_shape = (3, 3)
    
    model = Sequential()
    model.add(Conv2D(block1_units, kernel_size=kernel_shape,
                     activation='relu', input_shape=input_shape_img,
                     name='input_layer'))
    model.add(Conv2D(block1_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(block2_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(block2_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax',
                    name='logits_layer'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "vgg_2M_256FCU"
    return (model, model_name)


## vgg_1m_128fcu
def build_vgg_1M_128FCU():
    input_shape_img = deep_utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)

    block1_units = 48
    kernel_shape = (3, 3)
        
    model = Sequential()
    model.add(Conv2D(block1_units, kernel_size=kernel_shape,
                     activation='relu', input_shape=input_shape_img,
                     name='input_layer'))
    model.add(Conv2D(block1_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax',
                    name='logits_layer'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "vgg_1M_128FCU"
    return (model, model_name)

## vgg_3m_64fcu
def build_vgg_3M_64FCU():
    input_shape_img = deep_utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)

    block1_units = 48
    block2_units = 60
    block3_units = 72
    kernel_shape = (3, 3)
    
    model = Sequential()
    model.add(Conv2D(block1_units, kernel_size=kernel_shape,
                     activation='relu', input_shape=input_shape_img,
                     name='input_layer'))
    model.add(Conv2D(block1_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(block2_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(block2_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(block3_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(block3_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax',
                    name='logits_layer'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "vgg_3M_64FCU"
    return (model, model_name)


## vgg_4m_128fcu
def build_vgg_4M_128FCU():
    input_shape_img = deep_utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)

    block1_units = 48
    block2_units = 60
    block3_units = 72
    block4_units = 84
    kernel_shape = (3, 3)
    
    model = Sequential()
    model.add(Conv2D(block1_units, kernel_size=kernel_shape,
                     activation='relu', input_shape=input_shape_img,
                     name='input_layer'))
    model.add(Conv2D(block1_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(block2_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(block2_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(block3_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(block3_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(block3_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(block3_units, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax',
                    name='logits_layer'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "vgg_4M_128FCU"
    return (model, model_name)
