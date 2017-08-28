#-*-coding=UTF-8-*-
'''
tf.nn.dropout
env tf  = 0.11.
python 2.7.12
'''
import tensorflow as tf
# Usually the input would be output from a previous layer
# and not an image directly.
# 通常情况下，所谓的输入往往是来自于上一层的输出，而不是一个直接的图像
batch_size=1
input_height=3
input_width=3
input_channels=1

layer_input=tf.constant([
    [
        [[1.0], [0.2], [1.5]],
        [[0.1], [1.2], [1.4]],
        [[1.1], [0.4], [0.4]]
    ]
])
# The strides will look at the entire input by using the image_height and image_width
kernel=[batch_size, input_height, input_width, input_channels]
strides=[1,1,1,1]
max_pool=tf.nn.max_pool(layer_input, kernel, strides,"VALID" )
sess = tf.InteractiveSession()
print(sess.run(max_pool))