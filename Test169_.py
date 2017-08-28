#-*-coding=UTF-8-*-
'''
关于  tanh
env tf  = 0.11.0
python 2.7.12
'''

import tensorflow as tf

features = tf.to_float(tf.range(-1,3))

sess = tf.InteractiveSession() # 创建sess 对象
print('features =')
print(sess.run(features))
print('sigoid...')
print(sess.run([features, tf.tanh(features)]))