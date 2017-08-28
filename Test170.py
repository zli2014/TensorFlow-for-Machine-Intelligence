#-*-coding=UTF-8-*-
'''
tf.nn.dropout
env tf  = 0.11.
python 2.7.12
'''
import tensorflow as tf
features = tf.constant([-0.1, 0.0, 0.1, 0.2])
# Note, the output should be different on almost ever execution.
# Your numbers won't match this output
sess = tf.InteractiveSession()
print('the input and outputs is...')
print(sess.run([features , tf.nn.dropout(features, keep_prob=0.5)]))

