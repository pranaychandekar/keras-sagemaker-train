#!/usr/bin/env python3
"""
This file contains the model definition and the training logic.

:Author: Pranay Chandekar
"""
import os
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout

from constants import (
    MODEL_PATH,
    BATCH_SIZE,
    EPOCHS,
    NUM_CLASSES,
    X_TRAIN,
    Y_TRAIN,
    X_TEST,
    Y_TEST,
)


class Model:
    """
    This class contains the model definition and training logic.
    """

    def __init__(self, model_name: str, hyper_parameters: dict):
        self.model_name = model_name
        self.model = self.get_model_definition()
        self.hyper_parameters = hyper_parameters

    @staticmethod
    def get_model_definition():
        """
        This method contains the model definition.

        :return: The model
        """
        model = Sequential()
        model.add(Dense(512, activation="relu", input_shape=(784,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(NUM_CLASSES, activation="softmax"))
        return model

    def compile_model(self):
        """
        This method compiles the model and prints the model summary.
        """
        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.RMSprop(),
            metrics=["accuracy"],
        )
        self.model.summary()

    def train_model(self, processed_data: dict):
        """
        This method trains the model.
        """
        self.model.fit(
            x=processed_data[X_TRAIN],
            y=processed_data[Y_TRAIN],
            batch_size=self.hyper_parameters[BATCH_SIZE],
            epochs=self.hyper_parameters[EPOCHS],
            verbose=1,
            validation_data=(processed_data[X_TEST], processed_data[Y_TEST]),
        )

    def run(self, processed_data: dict):
        """
        This method calls all the other methods to perform model training, testing and saving.
        """
        print("\nStarting the model training")
        self.compile_model()
        self.train_model(processed_data)
        print("\nFinished training the model.")

        score = self.model.evaluate(
            processed_data[X_TEST], processed_data[Y_TEST], verbose=0
        )
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        print("\nSaving the model.")
        with open(
            os.path.join(MODEL_PATH, self.model_name + "_architecture.json"), "w"
        ) as architecture_file:
            architecture_file.write(self.model.to_json())

        self.model.save(os.path.join(MODEL_PATH, self.model_name + ".h5"))
        print("Finished saving the model.")
