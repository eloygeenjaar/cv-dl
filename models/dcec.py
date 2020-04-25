import keras
import numpy as np
from typing import Tuple

class DCEC(keras.Model):

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (28, 28, 1),
                 latent_space: Tuple[int, int, int] = (32, 64, 128),
                 kernels: Tuple[int, int, int] = (5, 5, 3),
                 strides: Tuple[int, int, int] = (2, 2, 2),
                 regularization: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                 use_dropout: bool = False,
                 use_batchnorm: bool = False,
                 dropout_percentages: Tuple[float, float, float] = None):
        super(DCEC, self).__init__(name='dcec')

        self.conv1 = keras.layers.Conv2D(filters=latent_space[0],
                                         kernel_size=kernels[0],
                                         stride=strides[0],
                                         padding='same',
                                         activation='relu',
                                         name='conv1',
                                         input_shape=input_shape)
        if use_batchnorm:
            self.batch_norm1 = keras.layers.BatchNormalization(axis=1)
        if use_dropout:
            self.dropout1 = keras.layers.Dropout(dropout_percentages[0])
        self.conv2 = keras.layers.Conv2D(filters=latent_space[1],
                                         kernel_size=kernels[1],
                                         strides=strides[1],
                                         padding='same',
                                         activation='relu',
                                         name='conv2')
        if use_batchnorm:
            self.batch_norm1 = keras.layers.BatchNormalization(axis=1)
        if use_dropout:
            self.dropout1 = keras.layers.Dropout(dropout_percentages[1])
        self.conv3 = keras.layers.Conv2D(filters=latent_space[2],
                                         kernel_size=kernels[2],
                                         strides=strides[2],
                                         padding='same',
                                         activation='relu',
                                         name='conv3',
                                         activity_regularizer=keras.regularizers.l2(regularization))
        if use_batchnorm:
            self.batch_norm1 = keras.layers.BatchNormalization(axis=1)
        if use_dropout:
            self.dropout1 = keras.layers.Dropout(dropout_percentages[2])





