#!/usr/bin/env python3
"""
This file contains all the global constants used by the project.

:Author: Pranay Chandekar
"""

import os

CONTAINER_PREFIX = "/opt/ml/"

INPUT_PATH = CONTAINER_PREFIX + "input/data"
OUTPUT_PATH = os.path.join(CONTAINER_PREFIX, "output")
MODEL_PATH = os.path.join(CONTAINER_PREFIX, "model")
PARAM_PATH = os.path.join(CONTAINER_PREFIX, "input/config/hyperparameters.json")

# This algorithm has a single channel of input data called 'training'. Since we run in
# File mode, the input files are copied to the directory specified here.
CHANNEL_NAME = "training"
TRAINING_PATH = os.path.join(INPUT_PATH, CHANNEL_NAME)

NUM_CLASSES = 10

X_TRAIN = "x_train"
Y_TRAIN = "y_train"
X_TEST = "x_test"
Y_TEST = "y_test"

BATCH_SIZE = "batch_size"
EPOCHS = "epochs"

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 1
