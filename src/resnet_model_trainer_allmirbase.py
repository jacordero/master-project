#!/usr/bin/env python

from __future__ import print_function
from datetime import date
import time

import keras
import numpy as np

from utils.data_loader import DataLoaderAllmirbase
import model_generators.resnet_model_generator as model_generator
import utils.deepmirna_utils as deep_utils

# Training parameters
EPOCHS = 100
BATCH_SIZE = 128
SAVE_MODEL = True

def train(model, train_datasets, test_datasets, model_name):
    # load datasets

    train_data = train_datasets[0]
    train_labels = train_datasets[1]
    train_names = train_datasets[2]

    test_data = test_datasets[0]
    test_labels = test_datasets[1]
    test_names = test_datasets[2]
            
    # train model
    model.fit(train_data, train_labels,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(test_data, test_labels),
              shuffle=True)

    # save trained model
    if SAVE_MODEL:
        deep_utils.create_directory("../models/")
        model_filename = "../models/allmirbase_" + model_name + ".h5"    
        model.save(model_filename)

    scores = model.evaluate(test_data, test_labels, verbose=1)
    return scores


# filters: 20, 24, 28, 32
NUM_FILTERS = 28
if __name__ == '__main__':
    start = time.time()

    script_name = "resnet_model_{}cl_trainer_allmirbase.py".format(NUM_FILTERS)
    print("******************************************************")
    print("Running: {}".format(script_name))
    print("Date: {}".format(str(date.today())))
    print("******************************************************")

    dataset = "allmirbase"
    loader = DataLoaderAllmirbase()
    train_datasets = loader.load_train_datasets()
    test_datasets = loader.load_test_datasets()
    
    model, model_name = model_generator.build_model_one_module(NUM_FILTERS)
    #print(model_name)
    scores = train(model, train_datasets, test_datasets, model_name)
    print("Results:{}, {}, {:.4f}".format(model_name, dataset, scores[1]))
    
    model, model_name = model_generator.build_model_two_modules(NUM_FILTERS)
    #print(model_name)
    scores = train(model, train_datasets, test_datasets, model_name)
    print("Results:{}, {}, {:.4f}".format(model_name, dataset, scores[1]))

    model, model_name = model_generator.build_model_three_modules(NUM_FILTERS)
    #print(model_name)
    scores = train(model, train_datasets, test_datasets, model_name)
    print("Results:{}, {}, {:.4f}".format(model_name, dataset, scores[1]))

    model, model_name = model_generator.build_model_four_modules(NUM_FILTERS)
    #print(model_name)
    scores = train(model, train_datasets, test_datasets, model_name)
    print("Results:{}, {}, {:.4f}".format(model_name, dataset, scores[1]))

    end = time.time()
    elapsed_time = (end - start) / 60
    print("\nTask finished in {} minutes!!".format(elapsed_time))
