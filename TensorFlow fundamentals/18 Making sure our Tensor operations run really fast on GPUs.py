import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

Physical_devices = tf.config.list_physical_devices("GPU")
print("Num GPUs:", len(Physical_devices))
print(Physical_devices)

# IF WE HAVE CUDA-ENABLED GPU, TENSORFLOW WILL AUTOMATICALLY USE IT WHENEVER NECESSARY.
