import tensorflow as tf

# BASIC OPERATIONS

# ADDING VALUES USING ADDITION OPERATOR
tensor = tf.constant([[1, 2],
                      [3, 4]])
print(tensor)
print(tensor + 10)  # ORIGINAL TENSOR WILL REMAIN UNCHANGED UNTIL IT IS ASSIGNED WITH THE INCREMENTED VALUES

# MULTIPLICATION
print(tensor * 10)

# SUBTRACTION
print(tensor - 1)

# WE CAN USE TENSORFLOW BUILT-IN FUNCTION TOO
print(tf.multiply(tensor, 10))

# PRACTICE WITH MUTLIPLE DIMENSIONS