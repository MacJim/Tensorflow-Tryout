"""
Source: https://www.tensorflow.org/guide/data

TODO: This file is unfinished.
"""

import os
import random

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf
import numpy as np


# MARK: - Constants
CLASS_NAMES = None


# MARK: - Functions
def _get_label_from_flower_path(absolute_filename: str) -> tf.Tensor:
    parts = absolute_filename.split(os.path.sep)


def test_flowers(val_split=0.2):
    flower_root = tf.keras.utils.get_file("flower_photos", "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz", untar=True)
    print(f"Flower root: {flower_root}")

    filenames = []
    flower_dir_names = os.listdir(flower_root)
    flower_dir_names = [os.path.join(flower_root, f) for f in flower_dir_names]
    flower_dir_names = [f for f in flower_dir_names if os.path.isdir(f)]
    print(f"Flower dir names: {flower_dir_names}")

    for dir_name in flower_dir_names:
        current_filenames = os.listdir(dir_name)
        current_filenames = [f for f in current_filenames if f.endswith(".jpg")]
        current_filenames = [os.path.join(dir_name, f) for f in current_filenames]
        filenames += current_filenames

    print(f"Flower files count: {len(filenames)}")    # 3670
    random.shuffle(filenames)

    filenames_dataset = tf.data.TextLineDataset(filenames)
    print(f"Filenames dataset: element specs: {filenames_dataset.element_spec}, len: {len(filenames_dataset)}")
    print(filenames_dataset[0])

    # Split into train and validation
    val_size = int(len(filenames) * 0.2)
    train_filenames = filenames[:val_size]
    val_filenames = filenames[val_size:]


# MARK: - Main
if (__name__ == "__main__"):
    # MARK: Switch to current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(f"Working directory: {os.getcwd()}")

    test_flowers()