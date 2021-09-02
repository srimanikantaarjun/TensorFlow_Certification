import tensorflow as tf
import numpy as np

# CREATE TENSORS WITH ONES (SIMILAR TO numpy.ones())
ones_tensor = tf.ones(shape = [3, 4], dtype = float)
print("Ones Tensor: \n", ones_tensor, "\n")

# CREATE TENSORS WITH ZEROS (SIMILAR TO numpy.zeros())
zeros_tensor = tf.zeros(shape = (3, 4), dtype = float)
print("Zeros Tensor: \n", zeros_tensor, "\n")

# WE CAN ALSO CREATE NUMPY ARRAYS INTO TENSORFLOW TENSORS
# THE MAIN DIFF BETWEEN NUMPY ARRAYS AND TENSORFLOW TENSORS IS THAT TENSORS CAN BE RUN ON A GPU
# (MUCH FASTER FOR NUMERICAL COMPUTING)

# X = tf.constant(some_matrix)  UPPERCASE FOR MATRIX OR TENSOR
# y = tf.constant(some_vector)  LOWERCASE FOR VECTOR

# https://numpy.org/doc/stable/reference/generated/numpy.arange.html
numpy_array = np.arange(start = 1, stop = 25, dtype = float)
print("Numpy array: \n", numpy_array, "\n")

# USE tf.constant() TO CONVERT A NUMPY ARRAY INTO TENSORFLOW TENSOR
A = tf.constant(value = numpy_array)
print("Numpy Array to Tensorflow Tensor: \n", A, "\n")

# TO GET A DIFF SHAPED TENSOR FROM ARRAY, THE DIMENSIONS SHOULD MATCH OF THE ORIGINAL ARRAY'S DIMENSIONS
A_2 = tf.constant(value = numpy_array, shape = [2, 3, 4])  # 2*3*4 (TENSOR.SHAPE) = 24 (ARRAY.SHAPE)
print("Numpy array into 3-D TensorFlow tensor: \n", A_2, "\n")
