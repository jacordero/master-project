#!/usr/bin/env python

from __future__ import print_function
from datetime import date
import time

import keras
import numpy as np

from utils.data_loader import DataLoaderHsa
import model_generators.resnet_model_generator as model_generator
import utils.deepmirna_utils as deep_utils

# Training parameters
PRETRAIN_EPOCHS = 1
TRAIN_EPOCHS = 2
BATCH_SIZE = 128
SAVE_MODEL = True


#TODO: review this
def train(model, pretrain_datasets, train_datasets,
          test_datasets, model_name):


    pretrain_data = pretrain_datasets[0]
    pretrain_labels = pretrain_datasets[1]
    pretrain_names = pretrain_datasets[2]
    
    train_data = train_datasets[0]
    train_labels = train_datasets[1]
    train_names = train_datasets[2]

    test_data = test_datasets[0]
    test_labels = test_datasets[1]
    test_names = test_datasets[2]
    
    model.fit(pretrain_data, pretrain_labels,
              batch_size=BATCH_SIZE,
              epochs=PRETRAIN_EPOCHS,
              shuffle=True)

    # train_model
    model.fit(train_data, train_labels,
              batch_size=BATCH_SIZE,
              epochs=TRAIN_EPOCHS,
              validation_data=(test_data, test_labels),
              shuffle=True)
    
    # save trained model
    if SAVE_MODEL:
        deep_utils.create_directory("../models")
        model_filename = "../models/hsa_" + model_name + ".h5"
        model.save(model_filename)

    scores = model.evaluate(test_data, test_labels, verbose=1)
    return scores


NUM_FILTERS = 28        
if __name__ == '__main__':
    start = time.time()

    script_name = "resnet_model_{}cl_trainer_hsa.py".format(NUM_FILTERS)
    print("******************************************************")
    print("Running: {}".format(script_name))
    print("Date: {}".format(str(date.today())))
    print("******************************************************")

    dataset = "hsa"
    loader = DataLoaderHsa()

    # pretrain model
    pretrain_datasets = loader.load_pretrain_datasets()
    train_datasets = loader.load_train_datasets() 
    test_datasets = loader.load_test_datasets()
    
    # build model
    model, model_name = model_generator.build_model_one_module(NUM_FILTERS)
    scores = train(model, pretrain_datasets, train_datasets, test_datasets, model_name)
    print("Results:{}, {}, {:.4f}".format(model_name, dataset, scores[1]))

    model, model_name = model_generator.build_model_two_modules(NUM_FILTERS)
    scores = train(model, pretrain_datasets, train_datasets, test_datasets, model_name)
    print("Results:{}, {}, {:.4f}".format(model_name, dataset, scores[1]))

    model, model_name = model_generator.build_model_three_modules(NUM_FILTERS)
    scores = train(model, pretrain_datasets, train_datasets, test_datasets, model_name)
    print("Results:{}, {}, {:.4f}".format(model_name, dataset, scores[1]))

    model, model_name = model_generator.build_model_four_modules(NUM_FILTERS)
    scores = train(model, pretrain_datasets, train_datasets, test_datasets, model_name)
    print("Results:{}, {}, {:.4f}".format(model_name, dataset, scores[1]))
    
    end = time.time()
    elapsed_time = (end - start) / 60
    print("\nTask finished in {} minutes!!".format(elapsed_time))
