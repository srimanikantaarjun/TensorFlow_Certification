import tensorflow as tf

# CREATE TENSOR WITH tf.Variable()
changeable_tensor = tf.Variable([1, 2])
unchangeable_tensor = tf.constant([1, 2])
print(changeable_tensor)
print(unchangeable_tensor)

# LET'S TRY CHANGING ONE OF OUR ELEMENTS IN CHANGEABLE TENSOR
print(changeable_tensor[0])
# changeable_tensor[0] = 4  # WE WILL GET A TYPEERROR

# HOW ABOUT WE TRY .assign()
changeable_tensor[0].assign(5)
print(changeable_tensor)

# LET'S TRY .assign() METHOD TO OUR UNCHANGEABLE TENSOR
# unchangeable_tensor[0].assign(6) WE WILL GET AN ATTRIBUTEERROR
# print(unchangeable_tensor)
