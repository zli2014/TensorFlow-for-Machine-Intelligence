#-*-coding=UTF-8-*-
'''
env tf  = 0.11.  python 2.7.12
'''
import tensorflow as tf
batch_size=1
input_height = 3
input_width = 3
input_channels = 1
layer_input = tf.constant([
         [
             [[1.0], [1.0], [1.0]],
             [[1.0], [0.5], [0.0]],
             [[0.0], [0.0], [0.0]]
         ]
])
# The strides will look at the entire input by using the image_height
# and image_width
kernel = [batch_size, input_height, input_width, input_channels]
max_pool = tf.nn.avg_pool(layer_input, kernel, [1, 1, 1, 1], "VALID")
sess=tf.InteractiveSession()
print(sess.run(max_pool))
