import tensorflow as tf

# PART I

# https://www.tensorflow.org/api_docs/python/tf/linalg/matmul
# SAME SHAPE MATRIX MULTIPLICATION
tensor = tf.constant([[1, 2],
                      [3, 4]])
multiplied_tensor = tf.linalg.matmul(tensor, tensor)
print(multiplied_tensor)

tensor2 = tf.constant([[[1, 2, 5],
                        [7, 2, 1],
                        [3, 3, 3]]])

tensor3 = tf.constant([[[3, 5],
                        [6, 7],
                        [1, 8]]])
multiplied_tensor2 = tf.linalg.matmul(tensor2, tensor3)
print(multiplied_tensor2)
# MATRIX MULTIPLICATION WITH PYTHON OPERATOR "@"
# tensor2 @ tensor3
# DIFFERENT SHAPE MATRIX MULTIPLICATION
# multiplied_tensor3 = tf.linalg.matmul(tensor3, tensor3) # ERROR: DIFFERENT SHAPE MATRICES
# print(multiplied_tensor3)
print("Tensor 3: ", tensor3)

# PART II - RESHAPING AND TRANSPOSE

# RESHAPING MATRIX
tensor4 = tf.reshape(tensor3, shape = [2, 3])
print("Tensor 4: \n", tensor4, "\n")
multiplied_tensor4 = tf.linalg.matmul(tensor3, tensor4)
print(multiplied_tensor4)

# TRANSPOSE MATRIX

transposed_tensor4 = tf.transpose(a = tensor3)
print("transposed tensor 4\n", transposed_tensor4, "\n")

# TRANSPOSE vs RESHAPE

# PART III - DOT PRODUCT

X = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]])
y = tf.constant([[7, 8],
                 [9, 10],
                 [11, 12]])
print("Shape of X and y: \n", X.shape, y.shape)
# https://www.tensorflow.org/api_docs/python/tf/tensordot
# Perform the dot product on X and Y (requires X or Y to be transposed)
print("Dot product of X and y:\n", tf.tensordot(tf.transpose(X), y, axes = 1))

# Perform matrix multiplication between X and y (transposed)
print("Matrix multiplication of X and transposed-y: \n", tf.matmul(a = X, b = tf.transpose(y)))

# Perform matrix multiplication between X and y (reshaped)
print("Matrix multiplication of X and reshaped y: \n", tf.matmul(a = X, b = tf.reshape(y, shape = [2, 3])))

# Check the values of y, transposed-y, and reshaped-y
print("\nOriginal y\n", y)
print("Transposed-y\n", tf.transpose(a = y))
print("y reshaped to (2, 3)\n", tf.reshape(tensor = y, shape = [2, 3]))

# WHEN PERFORMING MATRIX MULTIPLICATION ON TWO TENSORS AND ONE OF THE AXES DOESN'T LINE UP, YOU WILL TRANSPOSE
# (RATHER THAN RESHAPE) ONE OF THE TENSORS TO SATISFY THE MATRIX MULTIPLICATION RULES.
