"""
Source: https://www.tensorflow.org/guide/basic_training_loops
"""

import os

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# MARK: - Constants
# Settings
TRUE_W = 3.0
TRUE_B = 2.0
DATASET_SIZE = 1000    # In this simple script, this also serves as the batch size.
NUM_EPOCHS = 1000
LEARNING_RATE = 0.05

# Plot save locations
DATASET_PLOT_FILENAME = "13/dataset.png"
INITIAL_PREDICTION_PLOT_FILENAME = "13/initial_prediction.png"
# FINAL_PREDICTION_PLOT_FILENAME = "13/final_prediction.png"
TRAINING_VARIABLES_PLOT_FILENAME = "13/training_variables.png"


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


# MARK: - Loss
def get_l2_loss(target: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
    """
    L2 loss.
    """
    return tf.reduce_mean(tf.square(target - prediction))


# MARK: - Main
if (__name__ == "__main__"):
    # MARK: Switch to current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    # MARK: Create dataset
    x, target_y = get_dataset()
    plt.figure()
    plt.scatter(x, target_y, c="g")    # `c` is color
    # plt.show()
    plt.savefig(DATASET_PLOT_FILENAME)

    # MARK: Create model
    model = TestModel()
    print(f"Variables: {model.variables}")

    # MARK: Plot initial loss
    prediction_y = model(x)
    plt.figure()
    plt.scatter(x, prediction_y, c = "g")
    plt.savefig(INITIAL_PREDICTION_PLOT_FILENAME)

    initial_loss = get_l2_loss(target_y, prediction_y)
    print(f"Initial loss: {initial_loss}")

    # MARK: Train
    all_w = []
    all_b = []
    for epoch in range(NUM_EPOCHS):
        with tf.GradientTape() as tape:
            loss = get_l2_loss(target_y, model(x))

        print(f"{epoch + 1}. Loss: {loss}, w: {model.w.numpy()}, b: {model.b.numpy()}")

        dw, db = tape.gradient(loss, [model.w, model.b])
        model.w.assign_sub(LEARNING_RATE * dw)    # Subtract instead of add.
        model.b.assign_sub(LEARNING_RATE * db)

        all_w.append(model.w.numpy())
        all_b.append(model.b.numpy())

    plt.figure()
    plt.plot(range(NUM_EPOCHS), all_w, "r")
    plt.plot(range(NUM_EPOCHS), all_b, "g")
    plt.plot([TRUE_W] * NUM_EPOCHS, "r--")
    plt.plot([TRUE_B] * NUM_EPOCHS, "g--")
    plt.legend(["W", "b", "true W", "true b"])
    plt.savefig(TRAINING_VARIABLES_PLOT_FILENAME)
