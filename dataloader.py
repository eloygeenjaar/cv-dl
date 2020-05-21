from pathlib import Path
from typing import Tuple, List
from PIL import Image
import numpy as np
from time import perf_counter
import tensorflow as tf
import keras


class ChestXRayDataLoader(keras.utils.Sequence):
    """
    This is a custom Keras dataloader for the pneumonia Chest X-ray dataset
    Based on: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(self,
                 p: Path,
                 window_size: Tuple[int, int],
                 n_channels: int,
                 batch_size: int,
                 shuffle: bool = True,
                 extension: str = '.jpeg'):

        self.path: Path = p
        self.window_size: Tuple[int, int] = window_size
        self.n_channels: int = n_channels
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.extension: str = extension
        self.indexes: np.ndarray = np.array([])

        file_gen_healthy = self.path.glob(f'healthy/*{self.extension}')
        healthy_files = [f for f in file_gen_healthy if f.is_file()]
        print(f'{len(healthy_files)} images found in {self.path / "healthy"}')
        file_gen_pneumonia = self.path.glob(f'pneumonia/*{self.extension}')
        pneumonia_files = [f for f in file_gen_pneumonia if f.is_file()]
        print(f'{len(pneumonia_files)} images found in {self.path / "pneumonia"}')
        files = healthy_files + pneumonia_files
        temp_batch = []
        for f in files:
            img = Image.open(f)
            patches = self.__sliding_window_patches(np.array(img))
            temp_batch.append(patches)

        self.total_batch = np.vstack(temp_batch)
        self.total_batch = np.divide(self.total_batch, 255.0).astype(np.float32)
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.floor(self.total_batch.shape[0] / self.batch_size))

    def __getitem__(self, item) -> Tuple[np.ndarray, np.ndarray]:

        indexes = self.indexes[item * self.batch_size:(item + 1) * self.batch_size]

        batch = self.__data_generation(indexes)

        return (batch, batch)

    def on_epoch_end(self):
        """
        This function updates the indexes at the beginning as well as at the end of each epoch
        :return: None
        """
        self.indexes = np.arange(self.total_batch.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __sliding_window_patches(self,
                                 image: np.ndarray) -> np.ndarray:

        with tf.compat.v1.Session() as sess:
            image = tf.reshape(image, (1, *image.shape, 1))

            patches = tf.extract_image_patches(image,
                                               ksizes=[1, self.window_size[0], self.window_size[1], 1],
                                               strides=[1, self.window_size[0], self.window_size[1], 1],
                                               rates=[1, 1, 1, 1],
                                               padding='SAME')
            batch_size, num_windows_height, num_windows_width = patches.shape[0:3]
            new_batch_size = batch_size * num_windows_width * num_windows_height
            patches = tf.reshape(patches, (new_batch_size, *self.window_size, self.n_channels))

            return sess.run(patches)

    def __data_generation(self,
                          indexes: np.ndarray) -> np.ndarray:
        """
        This function produces the batches of data
        :param list_IDs_temp: The list of IDs that should be used for the batch
        :return: The new batch, with shape: (batch_size, window_height, window_width, n_channels)
        """

        batch = self.total_batch[indexes,].copy()

        return batch

# Example usage
"""
from dataloader import ChestXRayDataLoader
chest_path = Path('./example_images')

train_generator = ChestXRayDataLoader(p=chest_path / 'train',
                                      window_size=(256, 256),
                                      n_channels=1,
                                      batch_size=64)
validation_generator = ChestXRayDataLoader(p=chest_path / 'val',
                                      window_size=(256, 256),
                                      n_channels=1,
                                      batch_size=64)

unet.fit_generator(generator=train_generator,
                   validation_data=validation_generator,
                   epochs=10,
                   verbose=1,
                   use_multiprocessing=True,
                   workers=2)
"""

