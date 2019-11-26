#!/usr/bin/env python3
"""
This file contains the logic to read the hyper parameters.

:Author: Pranay Chandekar
"""
import json

from constants import PARAM_PATH, BATCH_SIZE, EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS


class HyperParameterReader:
    """
    This class contains the method to set the default hyper parameters and
    read the hyper parameters from the path.
    """

    @staticmethod
    def read_hyper_parameters():
        """
        This method reads the hyper parameter from the path.

        :return: The hyper parameters
        """
        hyper_parameters = HyperParameterReader.get_default_hyper_parameters()

        print("\nReading hyper parameters")

        with open(PARAM_PATH, "r") as hyper_parameters_file:
            parameters = json.load(hyper_parameters_file)

        if BATCH_SIZE in hyper_parameters:
            hyper_parameters[BATCH_SIZE] = int(parameters[BATCH_SIZE])

        if EPOCHS in hyper_parameters:
            hyper_parameters[EPOCHS] = int(parameters[EPOCHS])

        print("Finished reading the hyper parameters.")

        return hyper_parameters

    @staticmethod
    def get_default_hyper_parameters():
        """
        This method returns the default hyper parameters.

        :return: The default hyper parameters.
        """
        hyper_parameters = dict()
        hyper_parameters[BATCH_SIZE] = DEFAULT_BATCH_SIZE
        hyper_parameters[EPOCHS] = DEFAULT_EPOCHS
        return hyper_parameters
