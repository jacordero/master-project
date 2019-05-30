#!/usr/bin/env python

import os
import sys
import numpy as np

import deepmirna_utils as utils

NUM_CLASSES = 2

class DataLoaderHsa:

    def __init__(self):
        self.train_data_filename = "../datasets/hsa_train_data.npz"
        self.train_labels_filename = "../datasets/hsa_train_labels.npz"
        self.train_ids_filename = "../datasets/hsa_train_ids.npz"
        self.test_data_filename = "../datasets/hsa_test_data.npz"
        self.test_labels_filename = "../datasets/hsa_test_labels.npz"
        self.test_ids_filename = "../datasets/hsa_test_ids.npz"

        self.pretrain_data_filename = "../datasets/nonhsa_allmirbase_data.npz"
        self.pretrain_labels_filename = "../datasets/nonhsa_allmirbase_labels.npz"
        self.pretrain_ids_filename = "../datasets/nonhsa_allmirbase_names.npz"

    def load_pretrain_datasets(self):
        data = utils.load_image_data(self.pretrain_data_filename)
        labels = utils.load_labels(self.pretrain_labels_filename, NUM_CLASSES)
        ids = utils.load_ids(self.pretrain_ids_filename)
        return (data, labels, ids)

    def load_train_datasets(self):
        data = utils.load_image_data(self.train_data_filename)
        labels = utils.load_labels(self.train_labels_filename, NUM_CLASSES)
        ids = utils.load_ids(self.train_ids_filename)
        return (data, labels, ids)

    def load_test_datasets(self):
        data = utils.load_image_data(self.test_data_filename)
        labels = utils.load_labels(self.test_labels_filename, NUM_CLASSES)
        ids = utils.load_ids(self.test_ids_filename)
        return (data, labels, ids)
        

class DataLoaderAllmirbase:

    def __init__(self):
        self.train_data_filename = "../datasets/allmirbase_train_data.npz"
        self.train_labels_filename = "../datasets/allmirbase_train_labels.npz"
        self.train_ids_filename = "../datasets/allmirbase_train_ids.npz"
        self.test_data_filename = "../datasets/allmirbase_test_data.npz"
        self.test_labels_filename = "../datasets/allmirbase_test_labels.npz"
        self.test_ids_filename = "../datasets/allmirbase_test_ids.npz"

    def load_train_datasets(self):
        data = utils.load_image_data(self.train_data_filename)
        labels = utils.load_labels(self.train_labels_filename, NUM_CLASSES)
        ids = utils.load_ids(self.train_ids_filename)
        return (data, labels, ids)

    def load_test_datasets(self):
        data = utils.load_image_data(self.test_data_filename)
        labels = utils.load_labels(self.test_labels_filename, NUM_CLASSES)
        ids = utils.load_ids(self.test_ids_filename)
        return (data, labels, ids)
        
