#!/usr/bin/env python3
"""
This file contains the logic to read and pre-process the data.

:Author: Pranay Chandekar
"""

import os
import keras
import numpy as np

from constants import TRAINING_PATH, NUM_CLASSES, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST


class DataReader:
    """
    This class contains the methods to read and pre-process the data.
    """

    @staticmethod
    def get_data():
        """
        This method calls the other methods in an order to read, pre-process
        and split the data in train-test sets.

        :return: The processed data.
        """
        print("\nReading the data.")
        features, labels = DataReader.read_data()
        features, labels = DataReader.process_data(features, labels)
        processed_data = DataReader.test_train_split(features, labels)
        print("Finished data reading and pre-processing.")
        return processed_data

    @staticmethod
    def read_data():
        """
        This method reads the data from the data file.

        :return: The features and the labels.
        """
        labels = list()
        features = list()

        with open(os.path.join(TRAINING_PATH, "data_set")) as data_file:
            for data_point in data_file.readlines():
                temp = data_point.split(",")
                label = temp[0]
                feature = temp[1:]
                labels.append(label)
                features.append(feature)

        return features, labels

    @staticmethod
    def process_data(features: list, labels: list):
        """
        This method performs pre-processing of the data.

        :param features: The features of the data.
        :param labels: The target labels corresponding to features
        :return: The processed features and labels
        """
        features = np.asarray(features)
        features = features.astype("float32")
        features /= 255
        print("\nNumber of data samples: ", features.shape[0])

        labels = np.asarray(labels)
        labels = labels.astype(dtype=np.float32)
        labels = labels.astype(dtype=np.int)
        labels = keras.utils.to_categorical(labels, NUM_CLASSES)
        print("Number of data labels: ", labels.shape[0])

        return features, labels

    @staticmethod
    def test_train_split(features: list, labels: list):
        """
        This method performs the train-test split on the data.

        :param features: The features of the data.
        :param labels: The target labels corresponding to features
        :return: The train-test split
        """
        processed_data = dict()

        processed_data[X_TRAIN] = features[0:8000]
        processed_data[X_TEST] = features[8000:]
        processed_data[Y_TRAIN] = labels[0:8000]
        processed_data[Y_TEST] = labels[8000:]

        print(
            "\nNumber of training samples: ",
            (processed_data[X_TRAIN].shape, processed_data[Y_TRAIN].shape),
        )
        print(
            "Number of test samples: ",
            (processed_data[X_TEST].shape, processed_data[Y_TEST].shape),
        )

        return processed_data
