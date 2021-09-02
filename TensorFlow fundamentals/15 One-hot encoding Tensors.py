import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import numpy as np

# https://www.tensorflow.org/api_docs/python/tf/one_hot
some_list = [0, 1, 2, 3, 4, 5]
print("Sample list : ", some_list)
print("\n")
some_list_one_hot = tf.one_hot(indices = some_list, depth = 6)
print("One-hot encoding of Sample list : \n", some_list_one_hot)
print("\n")

# SPECIFY CUSTOM VALUES FOR ON-HOT ENCODING
print(tf.one_hot(some_list, depth = 6, on_value = "GPU", off_value = "CPU"))
