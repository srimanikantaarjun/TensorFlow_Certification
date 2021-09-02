import tensorflow as tf

# CREATE TENSORS FILLED WITH RANDOM VALUES

# CREATE 2 RANDOM TENSORS WITH SAME SIZE
random_tensor = tf.random.Generator.from_seed(42)  # set seed for reproducibility
random_tensor = random_tensor.normal(shape = (3, 2))
print(random_tensor)

random_tensor_2 = tf.random.Generator.from_seed(42)
random_tensor_2 = random_tensor_2.normal(shape = (3, 2))
print(random_tensor_2)

print(random_tensor == random_tensor_2)
# print(tf.math.equal(random_tensor, random_tensor_2))
