#!/usr/bin/env python

import os
import sys
import numpy as np
import pathlib

from os.path import normpath, join, abspath, dirname

import utils.deepmirna_utils as deep_utils

NUM_CLASSES = 2
data_dir = normpath(join(abspath(dirname(__file__)), "../../datasets/")) + "/"
indices_dir = data_dir + "kfold_indices/"

class DataLoaderHsa:

    def __init__(self):
        self.train_id_filenames = [
            indices_dir + "kfold_hsa_train_ids_0.npz",
            indices_dir + "kfold_hsa_train_ids_1.npz",
            indices_dir + "kfold_hsa_train_ids_2.npz",
            indices_dir + "kfold_hsa_train_ids_3.npz",
            indices_dir + "kfold_hsa_train_ids_4.npz",
            indices_dir + "kfold_hsa_train_ids_5.npz",
            indices_dir + "kfold_hsa_train_ids_6.npz",
            indices_dir + "kfold_hsa_train_ids_7.npz",
            indices_dir + "kfold_hsa_train_ids_8.npz",
            indices_dir + "kfold_hsa_train_ids_9.npz"
        ]

        self.test_id_filenames = [
            indices_dir + "kfold_hsa_test_ids_0.npz",
            indices_dir + "kfold_hsa_test_ids_1.npz",
            indices_dir + "kfold_hsa_test_ids_2.npz",
            indices_dir + "kfold_hsa_test_ids_3.npz",
            indices_dir + "kfold_hsa_test_ids_4.npz",
            indices_dir + "kfold_hsa_test_ids_5.npz",
            indices_dir + "kfold_hsa_test_ids_6.npz",
            indices_dir + "kfold_hsa_test_ids_7.npz",
            indices_dir + "kfold_hsa_test_ids_8.npz",
            indices_dir + "kfold_hsa_test_ids_9.npz"
        ]
        
        self.data_filename = data_dir + "hsa_data.npz"
        self.labels_filename = data_dir + "hsa_labels.npz"

        self.pretrain_data_filename = data_dir + "nonhsa_allmirbase_data.npz"
        self.pretrain_labels_filename = data_dir + "nonhsa_allmirbase_labels.npz"

        self.number_of_elements = self.compute_number_of_elements()
        
    def compute_number_of_elements(self):
        labels = deep_utils.load_labels(self.labels_filename, NUM_CLASSES)
        return labels.shape[0]

        
    def get_test_kfold_ids(self, fold_id):
        return deep_utils.load_ids(self.test_id_filenames[fold_id])
           
    def load_pretrain_datasets(self):
        data = deep_utils.load_image_data(self.pretrain_data_filename)
        labels = deep_utils.load_labels(self.pretrain_labels_filename, NUM_CLASSES)
        return (data, labels)

    def load_train_datasets(self, fold_id):
        ids = deep_utils.load_ids(self.train_id_filenames[fold_id])
        
        data = deep_utils.load_image_data(self.data_filename)
        data = np.take(data, ids, axis=0)

        labels = deep_utils.load_labels(self.labels_filename, NUM_CLASSES)
        labels = np.take(labels, ids, axis=0)
        return (data, labels)

    def load_test_datasets(self, fold_id):
        ids = deep_utils.load_ids(self.test_id_filenames[fold_id])
        
        data = deep_utils.load_image_data(self.data_filename)
        data = np.take(data, ids, axis=0)
        
        labels = deep_utils.load_labels(self.labels_filename, NUM_CLASSES)
        labels = np.take(labels, ids, axis=0)
        
        return (data, labels)

    
        
class DataLoaderAllmirbase:

    def __init__(self):
        self.train_id_filenames = [
            indices_dir + "kfold_allmirbase_train_ids_0.npz",
            indices_dir + "kfold_allmirbase_train_ids_1.npz",
            indices_dir + "kfold_allmirbase_train_ids_2.npz",
            indices_dir + "kfold_allmirbase_train_ids_3.npz",
            indices_dir + "kfold_allmirbase_train_ids_4.npz",
            indices_dir + "kfold_allmirbase_train_ids_5.npz",
            indices_dir + "kfold_allmirbase_train_ids_6.npz",
            indices_dir + "kfold_allmirbase_train_ids_7.npz",
            indices_dir + "kfold_allmirbase_train_ids_8.npz",
            indices_dir + "kfold_allmirbase_train_ids_9.npz"
        ]

        self.test_id_filenames = [
            indices_dir + "kfold_allmirbase_test_ids_0.npz",
            indices_dir + "kfold_allmirbase_test_ids_1.npz",
            indices_dir + "kfold_allmirbase_test_ids_2.npz",
            indices_dir + "kfold_allmirbase_test_ids_3.npz",
            indices_dir + "kfold_allmirbase_test_ids_4.npz",
            indices_dir + "kfold_allmirbase_test_ids_5.npz",
            indices_dir + "kfold_allmirbase_test_ids_6.npz",
            indices_dir + "kfold_allmirbase_test_ids_7.npz",
            indices_dir + "kfold_allmirbase_test_ids_8.npz",
            indices_dir + "kfold_allmirbase_test_ids_9.npz"
        ]
        
        self.data_filename = data_dir + "complete_allmirbase_data.npz"
        self.labels_filename = data_dir + "complete_allmirbase_labels.npz"
        self.number_of_elements = self.compute_number_of_elements()
    
        
    def compute_number_of_elements(self):
        
        labels = deep_utils.load_labels(self.labels_filename, NUM_CLASSES)
        return labels.shape[0]

    def load_train_datasets(self, fold_id):
        ids = deep_utils.load_ids(self.train_id_filenames[fold_id])
        
        data = deep_utils.load_image_data(self.data_filename)
        data = np.take(data, ids, axis=0)

        labels = deep_utils.load_labels(self.labels_filename, NUM_CLASSES)
        labels = np.take(labels, ids, axis=0)
        return (data, labels)

    def get_test_kfold_ids(self, fold_id):
        return deep_utils.load_ids(self.test_id_filenames[fold_id])
    
    def load_test_datasets(self, fold_id):
        ids = deep_utils.load_ids(self.test_id_filenames[fold_id])
        
        data = deep_utils.load_image_data(self.data_filename)
        data = np.take(data, ids, axis=0)
        
        labels = deep_utils.load_labels(self.labels_filename, NUM_CLASSES)
        labels = np.take(labels, ids, axis=0)
        
        return (data, labels)


