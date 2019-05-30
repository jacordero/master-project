#!/usr/bin/env python

import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from sklearn.model_selection import StratifiedKFold

import deepmirna_utils as utils

class DataGenerator:

    # initializer
    def __init__(self, kfolds):
        self.kfolds = kfolds        
        
        ## set default filenames
        self.data_filename = "../datasets/complete_allmirbase_data.npz"
        self.labels_filename = "../datasets/complete_allmirbase_labels.npz"
        self.categories_filename = "../datasets/complete_allmirbase_categories.npz"
        self.ids_filename = "../datasets/complete_allmirbase_names.npz"
        self.train_ids = []
        self.test_ids = []

        self.generate_kfolds()

    def generate_kfolds(self):
        print("[generate_kfolds] ...")

        data = np.load(self.data_filename)['arr_0']
        categories = np.load(self.categories_filename)['arr_0']
        
        kfold_generator = StratifiedKFold(n_splits=self.kfolds, shuffle=True)
        for train, test in kfold_generator.split(data, categories):   
            self.train_ids.append(train)
            self.test_ids.append(test)

        
    def get_partitions(self, fold_id):
        print("[generate_partitions] ...")
        if fold_id >= self.kfolds:
            raise("fold number: {} is greater than kfolds: {}".format(fold_id, self.kfolds))

        return (self.train_ids[fold_id], self.test_ids[fold_id])


if __name__ == '__main__':
    kfolds = 10
    data_generator = DataGenerator(kfolds)

    for fold_number in range(kfolds):
        print("Iteration: {}".format(fold_number))
        partitions = data_generator.get_partitions(fold_number)
        print("Train ids shape {}".format(partitions[0].shape))
        print("Test ids shape {}".format(partitions[1].shape))

        train_filename = "kfold_allmirbase_train_ids_" + str(fold_number) + ".npz"
        np.savez_compressed(train_filename, partitions[0])
        test_filename = "kfold_allmirbase_test_ids_" + str(fold_number) + ".npz"
        np.savez_compressed(test_filename, partitions[1])
