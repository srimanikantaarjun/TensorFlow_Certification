import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import numpy as np

# AGGREGATING TENSORS = CONDENSING THEM FROM MULTIPLE VALUES DOWN TO A SMALLER AMOUNT OF VALUES

# GET ABSOLUTE VALUES
A = tf.constant([-8, -18])
print("Tensor A : ", A)
print("\n")
B = tf.abs(x = A)
print("Absolute value of Tensor A is : ", B)
print("\n")

# LET'S GO THROUGH
# 1. GET THE MAXIMUM
# 2. GET THE MINIMUM
# 3. GET THE MEAN OF A TENSOR
# 4. GET THE SUM OF A TENSOR

# CREATE A RANDOM TENSOR

C = tf.constant(np.random.randint(0, 100, size = 50))
print("Randomly generated Tensor for testing out different aggregation methods : ", C)
print("\n")
print("Checking the attributes of the Tensor...")
print("Size of the Tensor : ", tf.size(C))
print("Shape of the Tensor : ", C.shape)
print("Dimensions of the Tensor : ", C.ndim)
print("\n")

# FIND A MINIMUM OF THE TENSOR
C_min = tf.reduce_min(C)
print("Minimum of the Tensor : ", C_min)
print("\n")
# FIND A MAXIMUM OF THE TENSOR
C_max = tf.reduce_max(C)
print("Maximum of the Tensor : ", C_max)
print("\n")
# FIND THE MEAN
C_mean = tf.reduce_mean(C)
print("Mean of the Tensor : ", C_mean)
print("\n")
# FIND THE SUM
C_sum = tf.reduce_sum(C)
print("Sum of the Tensor : ", C_sum)
print("\n")
