"""
Source: https://www.tensorflow.org/guide/intro_to_modules#saving_functions
"""

import os
import datetime

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf


# MARK: - Constants
VISUALIZATION_DIR = "visualizations"


# MARK: - Module definitions
class LazyDenseReLU (tf.Module):
    def __init__(self, out_features: int):
        super().__init__()

        self.is_built = False
        self.out_features = out_features
        self.b = tf.Variable(tf.zeros([out_features]), name="b")

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        if not self.is_built:
            self.w = tf.Variable(tf.random.normal([x.shape[-1], self.out_features]), name="w")
            self.is_built = True

        x = x @ self.w + self.b
        return tf.nn.relu(x)


class TestModule (tf.Module):
    def __init__(self):
        super().__init__()

        self.dense_1 = LazyDenseReLU(out_features=3)
        self.dense_2 = LazyDenseReLU(out_features=2)

    @tf.function    # The `__call__` method becomes an argument to `tf.function`.
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x


# MARK: - Visualization
def save_model_visualization(model: tf.Module, save_dir: str):
    """
    I saw a lot of deprecation warnings.
    Then why does the official guide still use such methods?
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(save_dir, str(stamp))
    writer = tf.summary.create_file_writer(save_dir)

    tf.summary.trace_on(graph=True, profiler=True)
    # Call only one tf.function when tracing.
    z = print(model(tf.constant([[2.0, 2.0, 2.0]])))
    with writer.as_default():
        tf.summary.trace_export(
            name="func_trace_11",
            step=0,
            profiler_outdir=save_dir)



# MARK: - Main
if (__name__ == "__main__"):
    # MARK: Switch to current dir 
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    # MARK: Create a model
    model = TestModule()
    save_model_visualization(model, VISUALIZATION_DIR)

    # MARK: Infer
    inputs = [tf.constant([[3.0, 3.0, 3.0]]), tf.constant([[[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]])]
    for input in inputs:
        output = model(input)
        print("Output:", output)    # Output shapes: (1, 2) and (1, 2, 2)
