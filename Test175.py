# -*-coding=UTF-8-*-
'''
env tf  = 0.11.  python 2.7.12
'''
import tensorflow as tf

image_input = tf.constant([
    [
        [[0., 0., 0.], [255., 255., 255.], [254., 0., 0.]],
        [[0., 191., 0.], [3., 108., 233.], [0., 191., 0.]],
        [[254., 0., 0.], [255., 255., 255.], [0., 0., 0.]]
    ]
])

conv2d = tf.contrib.layers.convolution2d(
    image_input,
    num_outputs=3,#num_output_channels=3,
    kernel_size=(1, 1),
    activation_fn=tf.nn.relu,
    stride=(1, 1),
    trainable=True)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
print(sess.run(conv2d))
