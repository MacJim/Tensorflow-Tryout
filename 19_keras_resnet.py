"""
Source: https://keras.io/api/applications/resnet/
"""
import os
import multiprocessing

from tensorflow import keras


# MARK: - Constants
SUMMARY_LINE_LEN = 132
ALTERNATIVE_INPUT_SHAPE = (256, 256, 3)
N_CLASSES = 128


# MARK: - Print summary
def test_resnet50_layers():
    model = keras.applications.ResNet50()
    print(f"Layers: count: {len(model.layers)}, type: {type(model.layers)}")    # Layers: count: 177, type: <class 'list'>
    for layer in model.layers:
        print(layer)


# MARK: - Write summary to file
def test_resnet50_default():
    """
    - When `include_top` is `True` (default) and `weights` is `"imagenet"` (default), input must has shape (224, 224, 3)
    """
    model = keras.applications.ResNet50(weights=None, input_shape=ALTERNATIVE_INPUT_SHAPE, classes=N_CLASSES)
    with open("19/resnet50_default.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'), line_length=SUMMARY_LINE_LEN)


def test_resnet50_include_top_false():
    """
    `include_top` is `False`.

    - No `Dense` at the end of the network
    - By default, no `GlobalAveragePooling2D` at the end of the network
      - This can be controlled using the `pooling` parameter
    - Can specify input shape (h, w, c)
      - Common sense: do not specify batch size
    """
    model = keras.applications.ResNet50(include_top=False, input_shape=ALTERNATIVE_INPUT_SHAPE)
    with open("19/resnet50_include_top_false.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'), line_length=SUMMARY_LINE_LEN)


# MARK: - Main
if (__name__ == "__main__"):
    # MARK: Switch to current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(f"Working directory: {os.getcwd()}")

    # MARK: Print summary
    # test_resnet50_layers()

    # MARK: Run some of the test functions concurrently
    test_functions = [test_resnet50_default, test_resnet50_include_top_false]
    processes = []

    for f in test_functions:
        p = multiprocessing.Process(target=f)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
