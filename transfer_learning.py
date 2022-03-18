# -*- encode: utf-8
""" TODO this file will be used as a basis for transfer learning
using command line.
"""
from typing import Tuple
import tensorflow as tf
from tensorflow import keras


def tf_train_test_split(
    folder: str,
    img_width: int = 299,
    img_height: int = 299,
    n_channels: int = 3,
    label_mode: str = 'binary',
    validation_split: int = 0.3,
    seed: int = 42,
    batch_size: int = 32) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """ TODO commenti
    """
    train = keras.processing.image_dataset_from_directory(
        folder,
        image_size=(img_width, img_height, n_channels),
        validation_split=validation_split,
        label_mode=label_mode,
        subset='training',
        batch_size=batch_size,
        seed=seed)
    val = keras.processing.image_dataset_from_directory(
        folder,
        image_size=(img_width, img_height, n_channels),
        validation_split=validation_split,
        label_mode=label_mode,
        subset='validation',
        batch_size=batch_size,
        seed=seed)
    return train, val


class BaseNetwork():
    
    @property
    def base_model(self):
        return self.__base_model
    
    def transfer_learning():
        pass

    def fine_tuning():
        pass


class TlXception(BaseNetwork):
    
    def __init__(
        self,
        img_width: int = 299,
        img_height: int = 299,
        n_channels: int = 3) -> None:
        self.base_model = keras.applications.Xception(
            weights='imagenet',
            input_shape=(
                img_width,
                img_height,
                n_channels),
            include_top=False
        )
