#!/usr/bin/anaconda3/bin/python3

from __future__ import print_function
import os
import sys
sys.path.append(os.getcwd())
from datetime import date
import time

import keras
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import numpy as np

import deepmirna_utils as utils
import vgg_model_generator as model_generator
from data_loader import DataLoaderHsa

PRETRAIN_EPOCHS = 40
TRAIN_EPOCHS = 100
BATCH_SIZE = 128
SAVE_MODEL = True
CHECKPOINT_ENABLED = False

def train(model, model_name):

    loader = DataLoaderHsa()

    pretrain_data, pretrain_labels, pretrain_names = loader.load_pretrain_datasets()
    
    # configure pretraining callbacks
    #tensorboard_log_dir = "../tensorboard_logs/hsa_pretrain_" + model_name
    #utils.create_directory(tensorboard_log_dir)
    #tensorboard = TensorBoard(log_dir=tensorboard_log_dir)
    #callbacks = [tensorboard]

    # pretrain model
    model.fit(pretrain_data, pretrain_labels,
              batch_size=BATCH_SIZE, epochs=PRETRAIN_EPOCHS)

    utils.create_directory("./models")
    model_filename = "./models/hsa_pretrain_" + model_name + ".h5" 
    model.save(model_filename)
    
    train_data, train_labels, train_names = loader.load_train_datasets()
    test_data, test_labels, test_names = loader.load_test_datasets()

    # configure training callbacks
    #tensorboard_log_dir = "./tensorboard_logs/hsa_" + model_name
    #utils.create_directory(tensorboard_log_dir)
    #tensorboard = TrainValTensorBoard(log_dir=tensorboard_log_dir)
    #callbacks = [tensorboard]

    #if CHECKPOINT_ENABLED:
    #    filepath = "./models/checkpoint_models/hsa_" + model_name + ".{epoch:02d}.h5"
    #    checkpoint = ModelCheckpoint(filepath, monitor="val_loss", verbose=0, period=10)
    #    callbacks.append(checkpoint)

    
    # train model
    model.fit(train_data, train_labels,
              validation_data=(test_data, test_labels),
              batch_size=BATCH_SIZE, epochs=TRAIN_EPOCHS)

    if SAVE_MODEL:
        utils.create_directory("./models")
        model_filename = "./models/hsa_" + model_name + ".h5" 
        model.save(model_filename)

    # evaluate the model
    scores = model.evaluate(test_data, test_labels, verbose=1)
    return scores

if __name__ == '__main__':
    start = time.time()

    script_name = "vgg_model_trainer_hsa.py"
    print("******************************************************")
    print("Script: {}".format(script_name))
    print("Date: {}".format(str(date.today())))
    print("******************************************************")

    model, model_name = model_generator.build_model_3modules_2x2()
    scores = train(model, model_name)
    print("Results: {}, hsa, {:.4f}".format(model_name, scores[1]))

    model, model_name = model_generator.build_model_3modules_3x3()
    scores = train(model, model_name)
    print("Results: {}, hsa, {:.4f}".format(model_name, scores[1]))

    model, model_name = model_generator.build_model_3modules_5x5()
    scores = train(model, model_name)
    print("Results: {}, hsa, {:.4f}".format(model_name, scores[1]))

    model, model_name = model_generator.build_model_3modules_cascade()
    scores = train(model, model_name)
    print("Results: {}, hsa, {:.4f}".format(model_name, scores[1]))

    
    end = time.time()
    elapsed_time = (end - start) / 60
    print("\nTask finished in {} minutes!!".format(elapsed_time))
