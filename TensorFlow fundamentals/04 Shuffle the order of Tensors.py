import tensorflow as tf

# WHY SHUFFLE THE ORDER OF ELEMENTS IN A TENSOR?

# VALUABLE WHEN WE WANT TO SHUFFLE OUR DATA SO THAT THE ORIGINAL ORDER DOESN'Y AFFECT LEARNING

not_shuffled_tensor = tf.constant([[1, 2],
                                   [3, 4],
                                   [5, 6]])
print(not_shuffled_tensor)
print(not_shuffled_tensor.ndim)

# tf.random.shuffle()

shuffled_tensor = tf.random.shuffle(value = not_shuffled_tensor)  # SHUFFLED ALONG 1ST DIMENSION BY DEFAULT (ALONG ROWS)
print(shuffled_tensor)
# READ THE DOCUMENTATION ON RANDOM SEED GENERATION
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
# AND PRACTICE WRITING 5 RANDOM TENSORS AND SHUFFLE THEM
