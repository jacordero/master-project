#!/usr/bin/env python

from __future__ import print_function

import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
import deepmirna_utils as utils

BATCH_SIZE = 128
NUM_CLASSES = 2
IMG_ROWS, IMG_COLUMNS = 25, 100

def build_model_kernel_2x2():
    input_shape_img = utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)

    kernel_shape = (2, 2)
    num_classes = 2
    
    model = Sequential()
    model.add(Conv2D(48, kernel_size=kernel_shape, activation='relu',
                     padding='same', input_shape=input_shape_img))
    model.add(Conv2D(48, kernel_size=kernel_shape, activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=(1, 1),
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(16, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(8, kernel_size=(1, 1),
                     activation='relu', padding='same'))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "fcn_kernel_2x2"
    return (model, model_name)


def build_model_kernel_3x3():
    input_shape_img = utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)

    kernel_shape = (3, 3)
    num_classes = 2
        
    model = Sequential()
    model.add(Conv2D(48, kernel_size=kernel_shape, activation='relu',
                     padding='same', input_shape=input_shape_img))
    model.add(Conv2D(48, kernel_size=kernel_shape, activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=(1, 1),
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(16, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(8, kernel_size=(1, 1),
                     activation='relu', padding='same'))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "fcn_kernel_3x3"
    return (model, model_name)


def build_model_kernel_5x5():
    input_shape_img = utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)
    num_classes = 2
    kernel_shape = (5, 5)
    
    model = Sequential()
    model.add(Conv2D(48, kernel_size=kernel_shape, activation='relu',
                     padding='same', input_shape=input_shape_img))
    model.add(Conv2D(48, kernel_size=kernel_shape, activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=(1, 1),
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(16, kernel_size=kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(8, kernel_size=(1, 1),
                     activation='relu', padding='same'))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "fcn_kernel_5x5"
    return (model, model_name)

def build_model_kernel_cascade532():
    input_shape_img = utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)
    num_classes = 2
    large_kernel_shape = (5, 5)
    medium_kernel_shape = (3, 3)
    small_kernel_shape = (2, 2)
    
    model = Sequential()
    model.add(Conv2D(48, kernel_size=large_kernel_shape,
                     activation='relu',
                     padding='same', input_shape=input_shape_img))
    model.add(Conv2D(48, kernel_size=large_kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=medium_kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=medium_kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=(1, 1),
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, kernel_size=small_kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(16, kernel_size=small_kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(8, kernel_size=(1, 1),
                     activation='relu', padding='same'))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "fcn_kernel_cascade532"
    return (model, model_name)

def build_model_kernel_cascade753():
    input_shape_img = utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)
    num_classes = 2
    large_kernel_shape = (7, 7)
    medium_kernel_shape = (5, 5)
    small_kernel_shape = (3, 3)
    
    model = Sequential()
    model.add(Conv2D(48, kernel_size=large_kernel_shape,
                     activation='relu',
                     padding='same', input_shape=input_shape_img))
    model.add(Conv2D(48, kernel_size=large_kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=medium_kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=medium_kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=(1, 1),
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, kernel_size=small_kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(16, kernel_size=small_kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(8, kernel_size=(1, 1),
                     activation='relu', padding='same'))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "fcn_kernel_cascade753"
    return (model, model_name)
