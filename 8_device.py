import os
import time

# Disable Tensorflow's debug messages.
# Must be set before importing Tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # DEBUG, INFO, WARNING, ERROR: 0 ~ 3

import tensorflow as tf


def test_default():
    """
    Runs on nVIDIA GPU by default.

    Somehow this simple script used 7501MiB of my GPU memory.
    """
    for _ in range(10000):
        c = tf.range(6, dtype=tf.float32)
        c += 1.0
        
        a = tf.reshape(c, (2, 3))
        b = tf.reshape(c, (3, 2))

        result = a @ b
        print(result)


def test_force_cpu():
    with tf.device("CPU:0"):
        for _ in range(10000):
            c = tf.range(6, dtype=tf.float32)
            c += 1.0
            
            a = tf.reshape(c, (2, 3))
            b = tf.reshape(c, (3, 2))

            result = a @ b
            print(result)


def test_separate_devices():
    """
    Store variables on one device and do the computation on another.
    This introduces some delay.
    """
    with tf.device("CPU:0"):
        c = tf.range(6, dtype=tf.float32)
        c += 1.0
        
        a = tf.reshape(c, (2, 3))
        b = tf.reshape(c, (3, 2))

    with tf.device("GPU:0"):
        for _ in range(10000):
            result = a @ b
            print(result)


# MARK: - Main
if (__name__ == "__main__"):
    start_time_default = time.time()
    test_default()
    end_time_default = time.time()

    start_time_cpu = time.time()
    test_force_cpu()
    end_time_cpu = time.time()

    start_time_separate = time.time()
    test_separate_devices()
    end_time_separate = time.time()

    print(f"test_default: {end_time_default - start_time_default}s")    # 4.9617204666137695s
    print(f"test_force_cpu: {end_time_cpu - start_time_cpu}s")    # 3.045450210571289s
    print(f"test_separate_devices: {end_time_separate - start_time_separate}s")    # 1.9795055389404297s
