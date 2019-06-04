#!/usr/bin/env python

from __future__ import print_function


import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import deepmirna_utils as utils

BATCH_SIZE = 128
NUM_CLASSES = 2
IMG_ROWS, IMG_COLUMNS = 25, 100
DENSE_UNITS = 256

def build_model_3modules_2x2():
    input_shape_img = utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)

    block1_units = 48
    block2_units = 60
    block3_units = 72
    kernel_shape = (2, 2)
    
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

    model_name = "vgg_3modules_2x2_dense{}".format(DENSE_UNITS)
    return (model, model_name)


def build_model_3modules_3x3():
    input_shape_img = utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)

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
    model.add(Dense(DENSE_UNITS, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax',
                    name='logits_layer'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "vgg_3modules_3x3_dense{}".format(DENSE_UNITS)
    return (model, model_name)

def build_model_3modules_5x5():
    input_shape_img = utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)

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

    model_name = "vgg_3modules_5x5_dense{}".format(DENSE_UNITS)
    return (model, model_name)


def build_model_3modules_cascade532():
    input_shape_img = utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)

    block1_units = 48
    block2_units = 60
    block3_units = 72
    block1_kernel_shape = (5, 5)
    block2_kernel_shape = (3, 3)
    block3_kernel_shape = (2, 2)
    
    model = Sequential()
    model.add(Conv2D(block1_units, kernel_size=block1_kernel_shape,
                     activation='relu', input_shape=input_shape_img,
                     name='input_layer'))
    model.add(Conv2D(block1_units, kernel_size=block1_kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(block2_units, kernel_size=block2_kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(block2_units, kernel_size=block2_kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(block3_units, kernel_size=block3_kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(block3_units, kernel_size=block3_kernel_shape,
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

    model_name = "vgg_3modules_cascade532_dense{}".format(DENSE_UNITS)
    return (model, model_name)

def build_model_3modules_cascade753():
    input_shape_img = utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)

    block1_units = 48
    block2_units = 60
    block3_units = 72
    block1_kernel_shape = (7, 7)
    block2_kernel_shape = (5, 5)
    block3_kernel_shape = (3, 3)
    
    model = Sequential()
    model.add(Conv2D(block1_units, kernel_size=block1_kernel_shape,
                     activation='relu', input_shape=input_shape_img,
                     name='input_layer'))
    model.add(Conv2D(block1_units, kernel_size=block1_kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(block2_units, kernel_size=block2_kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(block2_units, kernel_size=block2_kernel_shape,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(block3_units, kernel_size=block3_kernel_shape,
                     activation='relu', padding='same'))
    model.add(Conv2D(block3_units, kernel_size=block3_kernel_shape,
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

    model_name = "vgg_3modules_cascade753_dense{}".format(DENSE_UNITS)
    return (model, model_name)
