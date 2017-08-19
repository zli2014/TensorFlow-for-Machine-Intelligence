#-*-coding:UTF-8-*-

import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

input_batch = tf.constant([   # frist input
    [
        [[0.0], [1.0]],
        [[2.0], [3.0]]
    ],
    [
        [[4.0], [5.0]],
        [[5.0], [6.0]]
    ]
])

kernel = tf.constant([
        [
            [[1.0,2.0]]
        ]
])


conv2d = tf.nn.conv2d(input_batch,kernel, strides=[1,1,1,1],padding='SAME')

print("print ....")
print(input_batch)
print(kernel)

print("print  via  sess.run(.....)")
print(sess.run(input_batch))
print(sess.run(kernel))
#print(sess.run(conv2d))

