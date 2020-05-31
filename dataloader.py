from pathlib import Path
from typing import Tuple, List
import numpy as np
import tensorflow as tf
import keras
import cv2
from tqdm import tqdm

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
                 n_patches_per_file: int,
                 shuffle: bool = True,
                 extension: str = '.jpeg'):

        self.path: Path = p
        self.window_size: Tuple[int, int] = window_size
        self.n_channels: int = n_channels
        self.batch_size: int = batch_size
        self.n_patches_per_file: int = n_patches_per_file
        self.shuffle: bool = shuffle
        self.extension: str = extension
        self.indexes: np.ndarray = np.array([])

        self.file_gen_healthy = self.path.glob(f'NORMAL/*{self.extension}')
        healthy_files = [f for f in self.file_gen_healthy if f.is_file()]
        print(f'{len(healthy_files)} images found in {self.path / "NORMAL"}')
        self.file_gen_pneumonia = self.path.glob(f'PNEUMONIA/*{self.extension}')
        pneumonia_files = [f for f in self.file_gen_pneumonia if f.is_file()]
        print(f'{len(pneumonia_files)} images found in {self.path / "PNEUMONIA"}')
        self.files = healthy_files + pneumonia_files
        self.targets = np.array([0 if f in healthy_files else 1 for f in self.files])
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, item) -> Tuple[np.ndarray, np.ndarray]:

        indexes = self.indexes[item * self.batch_size:(item + 1) * self.batch_size]

        batch = self.__data_generation(indexes)

        return batch, batch

    def on_epoch_end(self):
        """
        This function updates the indexes at the beginning as well as at the end of each epoch
        :return: None
        """
        self.indexes = np.arange(len(self.files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __sample_multinomial(self,
                           patch_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:

        rows, columns = patch_shape
        mid_ix_cols = (columns-1)/2
        mid_ix_rows = (rows-1)/2
        var_rows = np.sqrt(rows) / 2
        var_cols = np.sqrt(columns) / 2
        row_ix = np.random.normal(loc=mid_ix_rows, scale=var_rows, size=self.n_patches_per_file).astype(int)
        col_ix = np.random.normal(loc=mid_ix_cols, scale=var_cols, size=self.n_patches_per_file).astype(int)

        row_ix[row_ix < 0] = 0
        row_ix[row_ix >= rows] = rows - 1
        col_ix[col_ix < 0] = 0
        col_ix[col_ix >= columns] = columns -1

        return row_ix, col_ix

    def __sliding_window_patches(self,
                                 image: np.ndarray) -> np.ndarray:

            image = tf.reshape(image, (1, *image.shape, 1))

            patches = tf.image.extract_patches(image,
                                               sizes=[1, self.window_size[0], self.window_size[1], 1],
                                               strides=[1, self.window_size[0], self.window_size[1], 1],
                                               rates=[1, 1, 1, 1],
                                               padding='SAME')
            batch_size, num_windows_height, num_windows_width = patches.shape[0:3]
            new_batch_size = batch_size * num_windows_width * num_windows_height
            patches = tf.reshape(patches, (new_batch_size, *self.window_size, self.n_channels))

            return patches

    def __data_generation(self,
                          indexes: np.ndarray) -> tf.Tensor:
        """
        This function produces the batches of data
        :param indexes: The list of IDs that should be used for the batch
        :return: The new batch, with shape: (batch_size, window_height, window_width, n_channels)
        """
        tensor_list = []
        for i in range(self.batch_size):
            img = cv2.imread(str(self.files[indexes[i]]), cv2.IMREAD_GRAYSCALE)
            patches_shape = np.ceil(img.shape[0]/self.window_size[0]), np.ceil(img.shape[1]/self.window_size[1])
            row_ix, col_ix = self.__sample_multinomial(patches_shape)
            indices = tf.constant(list(np.array(row_ix*patches_shape[1] + col_ix, dtype=np.int32)))
            patches = self.__sliding_window_patches(np.array(img))
            selected_patches = tf.gather(patches, indices)
            tensor_list.append(selected_patches)

        batch = tf.cast(tf.concat(tensor_list, axis=0), tf.float32)
        return tf.divide(batch, 255.0)

    def get_classes(self) -> np.ndarray:
        """
        This function returns a numpy array, where 0s respond to healthy files and 1s to pneumonia cases.
        This can be used when plotting a t-SNE plot for example.
        :return: Array with length len(self.files), where True means a file / index is a healthy patient
        """
        return np.repeat(self.targets[self.indexes], self.n_patches_per_file)

# Example usage
"""
from pathlib import Path
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

