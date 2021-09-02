import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

C = tf.constant(np.random.randint(0, 100, size = 50))
print("Randomly generated Tensor for testing out different aggregation methods : ", C)
print("\n")
print("Checking the attributes of the Tensor...")
print("Size of the Tensor : ", tf.size(C))
print("Shape of the Tensor : ", C.shape)
print("Dimensions of the Tensor : ", C.ndim)
print("\n")

# FIND THE VARIANCE AND STANDARD DEVIATION OF THE TENSOR USING TENSORFLOW METHODS
#C_var = tfp.stats.variance(C)   # DON'T NEED TENSORFLOW PROBABILITY NECESSARILY
C_var = tf.math.reduce_variance(input_tensor = tf.cast(x = C, dtype = tf.float32))
print("Variance of the Tensor : ", C_var)
print("\n")

# https://www.tensorflow.org/api_docs/python/tf/math/reduce_std
C_std = tf.math.reduce_std(tf.cast(x = C, dtype = tf.float32))
print("Standard deviation of the Tensor : ", C_std)
print("\n")
