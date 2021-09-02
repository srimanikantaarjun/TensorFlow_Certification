import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf

# CREATE A NEW TENSOR WITH FLOAT32 AS DEFAULT DATA TYPE
A = tf.constant([1., 2.])
print("Tensor A as float32 : ", A)
print("Data type of tensor A is : ", A.dtype)
print("\n")

# CHANGE FROM FLOAT32 TO FLOAT16 (REDUCED PRECISION)
B = tf.cast(x = A, dtype = tf.float16)
print("Tensor B as float16 : ", B)
print("Data type of tensor B as float16 is : ", B.dtype)
print("\n")

# CREATE A NEW TENSOR WITH INT32 AS DEFAULT DATA TYPE
C = tf.constant([1, 2])
print("Tensor C as int32 : ", C)
print("Data type of tensor C is : ", C.dtype)
print("\n")

# CHANGE INT32 TO FLOAT32
D = tf.cast(x = C, dtype = tf.int16)
print("Tensor D as int16 : ", D)
print("Data type of tensor D is : ", D.dtype)
# https://www.tensorflow.org/guide/mixed_precision
