import keras
import numpy as np
from typing import Tuple
from sklearn.preprocessing import Normalizer
import tensorflow as tf

class DCEC(keras.models.Sequential):

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (28, 28, 1),
                 latent_space: Tuple[int, int, int] = (32, 64, 128, 10),
                 kernels: Tuple[int, int, int] = (5, 5, 3),
                 strides: Tuple[int, int, int] = (2, 2, 2),
                 regularization: Tuple[float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0),
                 use_dropout: bool = False,
                 use_batchnorm: bool = False,
                 dropout_percentages: Tuple[float, float, float, float] = None):
        super(DCEC, self).__init__(name='dcec')

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        # Encoding part

        self.inputs = keras.layers.Input(input_shape)

        self.conv1 = keras.layers.Conv2D(filters=latent_space[0],
                                         kernel_size=kernels[0],
                                         strides=strides[0],
                                         padding='same',
                                         activation='relu',
                                         name='conv1',
                                         activity_regularizer=keras.regularizers.l2(regularization[0]),
                                         data_format='channels_last')

        if use_batchnorm:
            self.batch_norm1 = keras.layers.BatchNormalization(axis=1)
        if use_dropout:
            self.dropout1 = keras.layers.Dropout(dropout_percentages[0])

        self.conv2 = keras.layers.Conv2D(filters=latent_space[1],
                                         kernel_size=kernels[1],
                                         strides=strides[1],
                                         padding='same',
                                         activation='relu',
                                         name='conv2',
                                         activity_regularizer=keras.regularizers.l2(regularization[1]),
                                         data_format='channels_last')

        if use_batchnorm:
            self.batch_norm2 = keras.layers.BatchNormalization(axis=1)
        if use_dropout:
            self.dropout2 = keras.layers.Dropout(dropout_percentages[1])
        self.conv3 = keras.layers.Conv2D(filters=latent_space[2],
                                         kernel_size=kernels[2],
                                         strides=strides[2],
                                         padding='valid',
                                         activation='relu',
                                         name='conv3',
                                         activity_regularizer=keras.regularizers.l2(regularization[2]),
                                         data_format='channels_last')

        if use_batchnorm:
            self.batch_norm3 = keras.layers.BatchNormalization(axis=1)
        if use_dropout:
            self.dropout3 = keras.layers.Dropout(dropout_percentages[2])

        self.flatten = keras.layers.Flatten()

        self.dense1 = keras.layers.Dense(units=latent_space[3],
                                         name='dense1',
                                         activity_regularizer=keras.regularizers.l2(regularization[3]))
        if use_batchnorm:
            self.batch_norm4 = keras.layers.BatchNormalization(axis=1)
        if use_dropout:
            self.dropout4 = keras.layers.Dropout(dropout_percentages[3])

        # Decoding part
        # This following line was adopted from: https://github.com/XifengGuo/DCEC/blob/master/ConvAE.py
        self.dense2 = keras.layers.Dense(units=latent_space[2]*int(input_shape[0]/8)*int(input_shape[0]/8),
                                         activation='relu')

        # This following line was adopted from: https://github.com/XifengGuo/DCEC/blob/master/ConvAE.py
        self.reshape = keras.layers.Reshape((int(input_shape[0]/8), int(input_shape[0]/8), latent_space[2]))

        self.deconv1 = keras.layers.Conv2DTranspose(filters=latent_space[1],
                                                    kernel_size=kernels[2],
                                                    strides=strides[1],
                                                    padding='valid',
                                                    activation='relu',
                                                    name='deconv1')
        self.deconv2 = keras.layers.Conv2DTranspose(filters=latent_space[0],
                                                    kernel_size=kernels[1],
                                                    strides=strides[0],
                                                    padding='same',
                                                    activation='relu',
                                                    name='deconv2')
        self.deconv3 = keras.layers.Conv2DTranspose(filters=latent_space[2],
                                                    kernel_size=kernels[0],
                                                    strides=strides[2],
                                                    padding='same',
                                                    name='deconv3')

    def call(self,
             inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        x = self.conv1(inputs)
        if self.use_batchnorm:
            x = self.batch_norm1(x)
        if self.use_dropout:
            x = self.dropout1(x)
        x = tf.keras.backend.l2_normalize(x, axis=0)
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.batch_norm2(x)
        if self.use_dropout:
            x = self.dropout2(x)
        x = tf.keras.backend.l2_normalize(x, axis=0)
        x = self.conv3(x)
        if self.use_batchnorm:
            x = self.batch_norm3(x)
        if self.use_dropout:
            x = self.dropout3(x)
        x = tf.keras.backend.l2_normalize(x, axis=0)
        x = self.flatten(x)
        x = self.dense1(x)
        x = tf.keras.backend.l2_normalize(x, axis=0)
        if self.use_batchnorm:
            x = self.batch_norm4(x)
        if self.use_dropout:
            x = self.dropout4(x)
        x = self.dense2(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x

