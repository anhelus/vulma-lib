import pandas as pd


def to_csv(history, outfile):
    hist_df = pd.DataFrame(history)
    with open(outfile, mode='w') as f:
        hist_df.to_csv(f)



from typing import Tuple
import numpy as np
from tensorflow import keras

def get_img_array(
    img_path: str,
    size: Tuple[int, int] = (224, 224)):
    """ Read and preprocess an image.

    Args:
        img_path: location of the image
        size: expected size of the image    
    """
    img = keras.preprocessing.image.load_img(
        img_path,
        target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    # Transform the array into a batch by adding a dimension
    array = np.expand_dims(array, axis=0)
    return array