# In this file, we are going to be covering some of the fundamental concepts of tensord using TensorFlow

# More specifically,

# 1. Intro to tensors
# 2. Getting info from tensors
# 3. Manipulating tensors
# 4. Tensors and Numpy
# 5. Using tf.function( a way to speed up your regular python functions)
# 6. Using GPUs with TensorFlow (or TPUs)
# 7. Exercises to try for yourself

# Import tensorflow
import tensorflow as tf
print(tf.__version__)

# Create Tensors with tf.constant()
scalar = tf.constant(7)
print(scalar)
# Check number of dimensions of a tensor
print(scalar.ndim)

# Create a vector
vector = tf.constant([1, 2, 3])
print(vector)
# Check the dimension of the vector
print(vector.ndim)

# Create a matrix
matrix = tf.constant([[1, 2, 3],
                     [4, 5, 6]])
print(matrix)
# Check the dimension of matrix
print(matrix.ndim)

# CREATE ANOTHER MATRIX
matrix2 = tf.constant([[1., 2., 3.],
                       [4., 5., 6.]], dtype = tf.float16)  # specify the data type with dtype parameter
print(matrix2)
# CHECK DIMENSIONS OF matrix2
print(matrix2.ndim)

# LET'S CREATE A TENSOR
tensor = tf.constant([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
print(tensor)
# DIMENSIONS OF TENSOR
print(tensor.ndim)

# What we've created so far:

# Scalar: a single number
# Vector: a number with direction
# Matrix: a 2-dimensional array of numbers
# Tensor: a n-dimensional array of numbers
# (n can be any integer, a 0-dimensional tensor is a scalar, a 1-dimensional tensor is a vector)
