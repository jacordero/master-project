#!/usr/bin/env python

from __future__ import print_function
from datetime import date
import time

import keras
import numpy as np

import utils.deepmirna_utils as deep_utils
import model_generators.fcn_model_generator as model_generator
from utils.data_loader import DataLoaderAllmirbase

EPOCHS = 100
BATCH_SIZE = 128
SAVE_MODEL = True

def train(model, model_name):

    # load data
    loader = DataLoaderAllmirbase()    
    train_data, train_labels, train_names = loader.load_train_datasets()
    test_data, test_labels, test_names = loader.load_test_datasets()

    # train model
    model.fit(train_data, train_labels,
              validation_data=(test_data, test_labels),
              batch_size=BATCH_SIZE, epochs=EPOCHS)

    # save model
    if SAVE_MODEL:
        deep_utils.create_directory("../models")
        model_filename = "../models/allmirbase_" + model_name + ".h5" 
        model.save(model_filename)

    # evaluate model
    scores = model.evaluate(test_data, test_labels, verbose=1)
    return scores



if __name__ == '__main__':
    start = time.time()

    script_name = "fcn_model_trainer_allmirbase.py"
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

    model, model_name = model_generator.build_model_kernel_cascade532()
    scores = train(model, model_name)
    print("Results: {}, allmirbase, {:.4f}".format(model_name, scores[1]))

    model, model_name = model_generator.build_model_kernel_cascade753()
    scores = train(model, model_name)
    print("Results: {}, allmirbase, {:.4f}".format(model_name, scores[1]))
    
    end = time.time()
    elapsed_time = (end - start) / 60
    print("\nTask finished in {} minutes!!".format(elapsed_time))
