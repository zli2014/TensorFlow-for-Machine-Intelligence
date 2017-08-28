#-*-coding=UTF-8-*-
'''
env tf  = 0.11.  python 2.7.12
'''
import tensorflow as tf

# Create a range of 3 floats.
 #  TensorShape([batch, image_height, image_width, image_channels])
layer_input = tf.constant([
         [[[1.]], [[ 2.]], [[ 3.]]]
     ])
lrn = tf.nn.local_response_normalization(layer_input)
sess=tf.InteractiveSession()
print(sess.run([layer_input, lrn]))