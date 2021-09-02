import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import numpy as np

# CREATE A TENSOR
tf.random.set_seed(42)
A = tf.random.uniform(shape = [50])
print("Tensor A : ", A)
print("\n")

# POSITIONAL MAXIMUM
A_argmax = tf.argmax(A)
print("Index of maximum value of a Tensor : ", A_argmax)
print("\n")
# INDEX ON OUR LARGEST VALUE POSITION (TO VERIFY)
print("The value at A_argmax : ", A[A_argmax])
print("\n")
# FIND THE MAX VALUE OF A (TO VERIFY)
print("The maximum value of Tensor A : ", tf.reduce_max(A))
print("\n")
# CHECK FOR EQUALITY
print("Value at A_argmax == Maximum value of Tensor A : ", A[A_argmax] == tf.reduce_max(A))
print("\n")

# POSITIONAL MINIMUM
assert isinstance(A, object)
A_argmin = tf.argmin(A)
print("Index of minimum value of a Tensor : ", A_argmin)
print("\n")
# INDEX ON OUR SMALLEST VALUE POSITION (TO VERIFY)
print("The value at A_argmin : ", A[A_argmin])
print("\n")
# FIND THE MIN VALUE OF A (TO VERIFY)
print("The minimum value of Tensor A : ", tf.reduce_min(A))
print("\n")
# CHECK FOR EQUALITY
print("Value at A_argmin == Minimum value of Tensor A : ", A[A_argmin] == tf.reduce_min(A))
print("\n")
