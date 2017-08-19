#-*-coding=UTF-8-*-
import tensorflow as tf

image_batch = tf.constant([
    [ # fist image
        [[0,255,0], [0,255,0], [0,255,0]],
        [[0,255,0], [0,255,0], [0,255,0]]
    ],
    [ # second image
        [[0,0,255], [0,0,255], [0,0,255]],
        [[0,0,255], [0,0,255], [0,0,255]]
    ]
])
print(image_batch.get_shape())
sess = tf.InteractiveSession()
print('image batch 的第一个纬度度的第一个元素...')
print(sess.run(image_batch)[0][0][0])