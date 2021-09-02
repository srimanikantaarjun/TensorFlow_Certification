import tensorflow as tf

# TENSOR ATTRIBUTES TO BE AWARE OF WHEN DEALING WITH TENSORS

# SHAPE - tensor.shape - THE LENGTH OF EACH OF THE DIMENSIONS OF A TENSOR
# RANK - tensor.ndim - THE NUMBER OF TENSOR DIMENSIONS, A SCALAR HAS RANK 0, A VECTOR HAS RANK 1, A MATRIX HAS RANK 2,
# A TENSOR HAS RANK N
# AXIS (OR) DIMENSION - tensor[0], tensor[:,1],... - DIMENSION OF TENSOR
# SIZE - tf.size(tensor - TOTAL NUMBER OF ITEMS IN A TENSOR

# CREATE TENSOR WITH RANK 4

rank_4_tensor = tf.zeros(shape = [2, 3, 4, 5])
print("rank_4_tensor: \n", rank_4_tensor, "\n")

print("Shape of the tensor: ", rank_4_tensor.shape)
print("Number of Dimensions of the Tensor (rank): ", rank_4_tensor.ndim)
print("Size of the tensor (Total number of elements in our tensor): ", tf.size(rank_4_tensor))
print("Data Type of every element: ", rank_4_tensor.dtype)
print("Elements along 0th axis ", rank_4_tensor.shape[0])
print("Elements along nth axis ", rank_4_tensor.shape[-1])
print("Size of the tensor (Total number of elements in our tensor): ", tf.size(rank_4_tensor).numpy())
