import tensorflow as tf

# TENSORS CAN BE INDEXED JUST LIKE PYTHON LISTS

# GET 1ST TWO ELEMENTS OF EACH DIMENSION
rank_4_tensor = tf.zeros(shape = [2, 3, 4, 5])
print("rank_4_tensor: \n", rank_4_tensor, "\n")
print("1st two elements of th tensor: \n", rank_4_tensor[:2, :2, :2, :2], "\n")

# GET 1ST ELEMENT FROM EACH INDEX
print("1st element from each dimension except nth dimension: \n", rank_4_tensor[:1, :1, :1], "\n")
print("1st element from each dimension except (n-1)th dimension: \n", rank_4_tensor[:1, :1, :, :1], "\n")
# GET LAST ITEM OF EACH OF OUR RANK 2 TENSOR
rank_2_tensor = tf.constant(value = ([[1, 2],
                                      [3, 4]]))
print("Rank 2 Tensor: \n", rank_2_tensor, "\n")
print("Last item of each of our rank 2 tensor: \n", rank_2_tensor[:, -1], "\n")

# ADDING EXTRA DIMENSION TO TENSOR
rank_3_tensor = rank_2_tensor[..., tf.newaxis]
print("After adding a dimension to our rank 2 tensor: \n", rank_3_tensor, "\n")

# ALTERNATIVE TO tf.newaxis()
rank_5_tensor = tf.expand_dims(input = rank_4_tensor, axis = -1)
print("After adding a dimension to our rank 4 tensor: \n", rank_5_tensor, "\n")
# https://www.tensorflow.org/api_docs/python/tf/expand_dims
