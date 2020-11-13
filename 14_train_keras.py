"""
Source: https://www.tensorflow.org/guide/basic_training_loops#the_same_solution_but_with_keras
"""

import os

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# Settings
TRUE_W = 3.0
TRUE_B = 2.0
DATASET_SIZE = 1000    # In this simple script, this also serves as the batch size.
NUM_EPOCHS = 500
LEARNING_RATE = 0.05


# MARK: - Dataset
def get_dataset():
    x = tf.random.normal(shape=(DATASET_SIZE,))
    noise = tf.random.normal(shape=(DATASET_SIZE,))
    y = x * TRUE_W + TRUE_B + noise

    return (x, y)


# MARK: - Models
class TestModel (keras.Model):
    """
    1D `w * x + b`.
    """
    def __init__(self):
        super().__init__()
        
        self.w = tf.Variable(6.0)
        self.b = tf.Variable(0.0)

    def call(self, x: tf.Tensor):
        return self.w * x + self.b


# MARK: - Main
if (__name__ == "__main__"):
    # MARK: Switch to current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    # MARK: Create dataset
    x, target_y = get_dataset()

    # MARK: Create & train model
    model = TestModel()
    model.compile(run_eagerly=True, optimizer=keras.optimizers.SGD(learning_rate=LEARNING_RATE), loss=keras.losses.mean_squared_error)
    model.fit(x, target_y, epochs=NUM_EPOCHS, batch_size=DATASET_SIZE)

    print(f"Final W: {model.w.numpy()}, b: {model.b.numpy()}")
