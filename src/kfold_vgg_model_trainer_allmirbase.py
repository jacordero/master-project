#!/usr/bin/env python

from __future__ import print_function
from datetime import date
import time

import keras
import numpy as np

from utils.kfold_data_loader import DataLoaderAllmirbase
import model_generators.final_vgg_model_generator as model_generator

EPOCHS = 100
BATCH_SIZE = 128

def train_one_fold(model, train_datasets, test_datasets):

    # load data
    train_data = train_datasets[0]
    train_labels = train_datasets[1]
    test_data = test_datasets[0]
    test_labels = test_datasets[1]

    # train model
    model.fit(train_data, train_labels,
              validation_data=(test_data, test_labels),
              batch_size=BATCH_SIZE, epochs=EPOCHS,
              verbose=2)

    return model.evaluate(test_data, test_labels)


def train_kfolds(data_loader, model_generator, model_type):
    kfolds = 10
    aggregated_predictions = np.zeros((data_loader.number_of_elements))
    acc_values = []
    
    for fold_id in range(kfolds):
        train_datasets = data_loader.load_train_datasets(fold_id)
        test_datasets = data_loader.load_test_datasets(fold_id)
        test_kfold_ids = data_loader.get_test_kfold_ids(fold_id)

        print('Iteration:{}'.format(fold_id))
        
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
    
        acc = train_one_fold(model, train_datasets, test_datasets)
        acc_values.append(acc)

    return acc_values



if __name__ == '__main__':
    script_name = "kfold_vgg_allmirbase.py"
    print("******************************************************")
    print("Running: {}".format(script_name))
    print("Date: {}".format(str(date.today())))
    print("******************************************************")
    start = time.time()
    
    data_loader = DataLoaderAllmirbase()
    
    # vgg_3M_256FCU
    acc_values = train_kfolds(data_loader, model_generator, 'vgg_3M_256FCU')
    acc_values_filename = "../kfold_acc_values/kfold_vgg_3M_256FCU_allmirbase.npz"
    np.savez_compressed(acc_values_filename, acc_values)
    
    # vgg_2M_256FCU
    acc_values = train_kfolds(data_loader, model_generator, 'vgg_2M_256FCU')
    acc_values_filename = "../kfold_acc_values/kfold_vgg_2M_256FCU_allmirbase.npz"
    np.savez_compressed(acc_values_filename, acc_values)

    # vgg_1M_128FCU
    acc_values = train_kfolds(data_loader, model_generator, 'vgg_1M_128FCU')
    acc_values_filename = "../kfold_acc_values/kfold_vgg_1M_128FCU_allmirbase.npz"
    np.savez_compressed(acc_values_filename, acc_values)
    
    end = time.time()
    elapsed_time = (end - start) / 60
    print("\nTask finished in {} minutes!!".format(elapsed_time))
