"""
https://www.tensorflow.org/guide/keras/sequential_model
"""

import os

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

# import update_cudnn_path
# update_cudnn_path.main()

import tensorflow as tf
from tensorflow import keras


def test_feature_extraction():
    """
    Get output(s) from a specific layer/specific layers.

    Although the parameter name `outputs` is plural, it actually accepts both lists or layer outputs.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=(250, 250, 3)),
            keras.layers.Conv2D(32, 5, strides=2, activation="relu"),
            keras.layers.Conv2D(32, 3, activation="relu", name="layer2"),
            keras.layers.Conv2D(32, 3, activation="relu"),
        ]
    )

    feature_extractor_1 = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])    # Extract from multiple layers.
    feature_extractor_2 = keras.Model(inputs=model.inputs, outputs=model.get_layer("layer2").output)    # Extract from a single layer.

    x = tf.ones((1, 250, 250, 3))
    feature_1 = feature_extractor_1(x)
    feature_2 = feature_extractor_2(x)

    print(f"Feature 1 type: {type(feature_1)}")    # list
    for feature in feature_1:
        print(f"Shape: {feature.shape}")    # (1, 123, 123, 32), (1, 121, 121, 32), (1, 119, 119, 32)
    
    print(f"Feature 2 type: {type(feature_2)}, shape: {feature_2.shape}")    # tensorflow.python.framework.ops.EagerTensor (1, 121, 121, 32)


# MARK: - Main
if (__name__ == "__main__"):
    # MARK: Switch to current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # print(1)
    # tf.load_library("/scratch/MJ/cuda/cudnn-10.0-v7.6.5.32/lib64/libcudnn.so.7")
    # print(2)
    # tf.load_library("/scratch/MJ/cuda/cudnn-10.0-v7.6.5.32/lib64/libcudnn.so.7")

    test_feature_extraction()
