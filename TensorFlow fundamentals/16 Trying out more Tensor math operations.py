import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import numpy as np

# CREATE A TENSOR
A = tf.range(1, 10)
print("Example Tensor : ", A)
print("\n")
# SQUARE THE TENSOR
A_squared = tf.square(A)
print("Example Tensor Squared : ", A_squared)
print("\n")
# SQUARE ROOT OF TENSOR
A_sqrt = tf.sqrt(tf.cast(x = A, dtype = tf.float32))  # tf.sqrt DOESN'T WORK ON dtype = int32
print("Square root of Tensor : ", A_sqrt)
print("\n")
# FIND LOG
A_log = tf.math.log(tf.cast(x = A, dtype = tf.float32))
print("Log values of Tensor : ", A_log)
print("\n")
