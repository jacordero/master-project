#!/usr/bin/env python

from __future__ import print_function

from datetime import date
import time

import keras
from keras.models import load_model

import numpy as np

import utils.deepmirna_utils as deep_utils
from utils.kfold_data_loader import DataLoaderHsa
import model_generators.final_vgg_model_generator as model_generator

PRETRAIN_EPOCHS = 40
TRAIN_EPOCHS = 100
BATCH_SIZE = 128

def pretrain(model, model_name, pretrain_datasets):

    pretrain_data = pretrain_datasets[0]
    pretrain_labels = pretrain_datasets[1]
    
    # pretrain model
    model.fit(pretrain_data, pretrain_labels,
              batch_size=BATCH_SIZE, epochs=PRETRAIN_EPOCHS,
              verbose=2)

    deep_utils.create_directory("../models")
    model_filename = "../models/kfold_hsa_pretrain_" + model_name + ".h5" 
    model.save(model_filename)

    return model_filename


def train_one_fold(model, train_datasets, test_datasets):

    # load data
    train_data = train_datasets[0]
    train_labels = train_datasets[1]
    test_data = test_datasets[0]
    test_labels = test_datasets[1]

    # train model
    model.fit(train_data, train_labels,
              validation_data=(test_data, test_labels),
              batch_size=BATCH_SIZE, epochs=TRAIN_EPOCHS,
              verbose=2)

    return model.evaluate(test_data, test_labels)




def train_kfolds(data_loader, model_generator, model_type):

    if model_type == 'vgg_3M_256FCU':
        model, model_name = model_generator.build_vgg_3M_256FCU()
    if model_type == 'vgg_2M_256FCU':
        model, model_name = model_generator.build_vgg_2M_256FCU()
    if model_type == 'vgg_1M_128FCU':
        model, model_name = model_generator.build_vgg_1M_128FCU()
    if model_type == 'vgg_3M_64FCU':
        model, model_name = model_generator.build_vgg_3M_64FCU()
    if model_type == 'vgg_4M_128FCU':
        model, model_name = model_generator.build_vgg_4M_128FCU()

    pretrain_datasets = data_loader.load_pretrain_datasets()    
    pretrained_model_filename = pretrain(model, model_name, pretrain_datasets)
    
    kfolds = 10
    aggregated_predictions = np.zeros((data_loader.number_of_elements))
    acc_values = []
    
    for fold_id in range(kfolds):
        print("Model: {},  iteration: {}".format(model_name, fold_id))

        train_datasets = data_loader.load_train_datasets(fold_id)
        test_datasets = data_loader.load_test_datasets(fold_id)
        test_kfold_ids = data_loader.get_test_kfold_ids(fold_id)
        
        pretrained_model = load_model(pretrained_model_filename)
        acc = train_one_fold(pretrained_model,
                             train_datasets,
                             test_datasets)

        acc_values.append(acc)
        #aggregated_predictions[test_kfold_ids] = predictions

    return np.asarray(acc_values)
    #return aggregated_predictions
    
    
if __name__ == '__main__':
    
    start = time.time()
    script_name = "kfold_vgg_hsa.py"
    print("******************************************************")
    print("Running: {}".format(script_name))
    print("Date: {}".format(str(date.today())))
    print("******************************************************")

    data_loader = DataLoaderHsa()

        # vgg_3M_256FCU
    acc_values = train_kfolds(data_loader, model_generator, 'vgg_3M_256FCU')
    acc_values_filename = "../kfold_acc_values/kfold_vgg_3M_256FCU_hsa.npz"
    np.savez_compressed(acc_values_filename, acc_values)
    
    # vgg_2M_256FCU
    acc_values = train_kfolds(data_loader, model_generator, 'vgg_2M_256FCU')
    acc_values_filename = "../kfold_acc_values/kfold_vgg_2M_256FCU_hsa.npz"
    np.savez_compressed(acc_values_filename, acc_values)

    # vgg_1M_128FCU
    acc_values = train_kfolds(data_loader, model_generator, 'vgg_1M_128FCU')
    acc_values_filename = "../kfold_acc_values/kfold_vgg_1M_128FCU_hsa.npz"
    np.savez_compressed(acc_values_filename, acc_values)
    
    
    end = time.time()
    elapsed_time = (end - start) / 60
    print("\nTask finished in {} minutes!!".format(elapsed_time))
