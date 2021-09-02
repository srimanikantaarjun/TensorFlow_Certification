import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import numpy as np

# CREATE A TENSOR DIRECTLY FROM A NUMPY ARRAY
A = tf.constant(np.array([2.0, 4.0, 6.0, 8.0]))
print("Sample Tensor : ", A)
print("\n")
# CONVERT THE TENSOR BACK INTO A NUMPY ARRAY
A_to_array = np.array(A)
print("Sample Tensor to Array : ", A_to_array)
print("Type of resulting array : ", type(A_to_array))
print("\n")
# CONVERT TENSOR INTO NUMPY ARRAY USING .numpy() METHOD
A_2_array = A.numpy()
print("Sample Tensor to array using .numpy() : ", A_2_array)
print("Type of resulting array : ", type(A_2_array))
