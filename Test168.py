#-*-coding=UTF-8-*-
'''
书中的第 168页中涉及的程序
'''
import tensorflow as tf

features = tf.range(-2, 3)

sess = tf.InteractiveSession()
# keep note of the value for native features
print(sess.run([features, tf.nn.relu(features)]))