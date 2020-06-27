import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import numpy as np
from typing import Tuple


class ChestXRayDataLoaderV2(keras.utils.Sequence):
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
        self.samples = []
        self.layout = (2,2)

        self.file_gen_healthy = self.path.glob(f'NORMAL/*{self.extension}')
        healthy_files = [f for f in self.file_gen_healthy if f.is_file()]
        print(f'{len(healthy_files)} images found in {self.path / "NORMAL"}')
        self.file_gen_pneumonia = self.path.glob(
            f'PNEUMONIA/*{self.extension}')
        pneumonia_files = [f for f in self.file_gen_pneumonia if f.is_file()]
        print(f'{len(pneumonia_files)} images found in {self.path / "PNEUMONIA"}')
        self.files = healthy_files + pneumonia_files

        self.targets = np.array(
            [0 if f in healthy_files else 1 for f in self.files])
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, item) -> Tuple[np.ndarray, np.ndarray]:

        indexes = self.indexes[item *
                               self.batch_size:(item + 1) * self.batch_size]

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

    def __process_path(self, path):
        """
        This function will preprocess a file img path.
        """
        img = tf.io.read_file(path)

        img = tf.io.decode_jpeg(img, channels=1, ratio=2,
                                name="resizeand reshape")

        img = tf.cast(img, tf.float32)

        return img / 255.0

    def __tile_image(self):
        """
        This function will create a tile grid for the gaussians.
        """
        tiles_x, tiles_y = self.layout

        centers_x = np.linspace(-1, 1, num=1+tiles_x*2)[1::2]
        centers_y = np.linspace(-1, 1, num=1+tiles_y*2)[1::2]
        
        centers_x = tf.transpose(tf.random.normal((1,tiles_x), mean=0, stddev=0.2) +  centers_x.T)
        centers_y = tf.transpose(tf.random.normal((1,tiles_y), mean=0, stddev=0.2) +  centers_y.T)
        
        offsets = tf.reshape(np.meshgrid(centers_x, centers_y), (2, -1))
        offsets = tf.transpose(offsets)

        return offsets
        
    def __grid_patches(self, img):
        """
        This function will generate patches based on a 2x2 grid
        :param img: The image tensor
        """
        shape = img.shape
        offsets = self.__tile_image()

        return tf.image.extract_glimpse(
            tf.tile(tf.expand_dims(img, 0), [offsets.shape[0], 1, 1, 1]), self.window_size, offsets=offsets, centered=True, normalized=True, noise='uniform', name="extract patch(es)"
        )

    def __data_generation(self, indexes: np.ndarray) -> tf.Tensor:
        """
        This function produces the batches of data
        :param indexes: The list of IDs that should be used for the batch
        :return: The new batch, with shape: (batch_size, window_height, window_width, n_channels)
        """
        tensor_list = []
        for i in range(self.batch_size):
            img = self.__process_path(str(self.files[indexes[i]]))
            tensor_list.append(self.__grid_patches(img))
        batch = tf.concat(tensor_list, axis=0)
        return batch

    def get_classes(self) -> np.ndarray:
        """
        This function returns a numpy array, where 0s respond to healthy files and 1s to pneumonia cases.
        This can be used when plotting a t-SNE plot for example.
        :return: Array with length len(self.files), where True means a file / index is a healthy patient
        """
        return np.repeat(self.targets[self.indexes], self.n_patches_per_file)

