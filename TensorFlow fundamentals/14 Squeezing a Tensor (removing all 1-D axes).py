import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import numpy as np

# CREATE A TENSOR
tf.random.set_seed(42)
A = tf.constant(tf.random.uniform(shape = [50]), shape = [1, 1, 1, 1, 50])
print("Created Tensor A : ", A)
print("\n")
print("Shape of the Tensor : ", A.shape)
print("\n")

A_squeezed = tf.squeeze(A)
print("Squeezed tensor A : ", A_squeezed)
print("\n")
print("Shape of the Squeezed Tensor : ", A_squeezed.shape)
print("\n")
