#!/usr/bin/anaconda3/bin/python3

from __future__ import print_function
import os
import sys
sys.path.append(os.getcwd())
from datetime import date
import time

import keras
import numpy as np
from keras.callbacks import ModelCheckpoint


import deepmirna_utils as utils
import fcn_model_generator as model_generator
from data_loader import DataLoaderAllmirbase



EPOCHS = 100
BATCH_SIZE = 128
SAVE_MODEL = True
CHECKPOINT_ENABLED = False


def train(model, model_name):

    # load data
    loader = DataLoaderAllmirbase()    
    train_data, train_labels, train_names = loader.load_train_datasets()
    test_data, test_labels, test_names = loader.load_test_datasets()

    # configure callbacks
    #tensorboard_log_dir = "./tensorboard_logs/allmirbase_" + model_name
    #utils.create_directory(tensorboard_log_dir)
    #tensorboard = TrainValTensorBoard(log_dir=tensorboard_log_dir)
    #callbacks = [tensorboard]
    
    #if CHECKPOINT_ENABLED:
    #    filepath = "./models/checkpoint_models/allmirbase_" + model_name + ".{epoch:02d}.h5"
    #    checkpoint = ModelCheckpoint(filepath, monitor="val_loss", verbose=0, period=10)
    #    callbacks.append(checkpoint)

    # train model
    model.fit(train_data, train_labels,
              validation_data=(test_data, test_labels),
              batch_size=BATCH_SIZE, epochs=EPOCHS)

    # save model
    if SAVE_MODEL:
        utils.create_directory("./models")
        model_filename = "./models/allmirbase_" + model_name + ".h5" 
        model.save(model_filename)

    # evaluate model
    scores = model.evaluate(test_data, test_labels, verbose=1)
    return scores



if __name__ == '__main__':
    start = time.time()

    script_name = "vgg_model_trainer_allmirbase.py"
    print("******************************************************")
    print("Script: {}".format(script_name))
    print("Date: {}".format(str(date.today())))
    print("******************************************************")

    
    model, model_name = model_generator.build_model_kernel_2x2()
    scores = train(model, model_name)
    print("Results: {}, allmirbase, {:.4f}".format(model_name, scores[1]))

    model, model_name = model_generator.build_model_kernel_3x3()
    scores = train(model, model_name)
    print("Results: {}, allmirbase, {:.4f}".format(model_name, scores[1]))

    model, model_name = model_generator.build_model_kernel_5x5()
    scores = train(model, model_name)
    print("Results: {}, allmirbase, {:.4f}".format(model_name, scores[1]))

    model, model_name = model_generator.build_model_kernel_cascade()
    scores = train(model, model_name)
    print("Results: {}, allmirbase, {:.4f}".format(model_name, scores[1]))
    
    end = time.time()
    elapsed_time = (end - start) / 60
    print("\nTask finished in {} minutes!!".format(elapsed_time))
