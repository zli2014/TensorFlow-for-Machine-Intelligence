# -*-coding=UTF-8-*-
'''
env tf  = 0.11.  python 2.7.12
'''
import tensorflow as tf

features = tf.constant([
    [[1.2],[3.4]]
])
# 此处与之前的有所不同  Attempting to use uninitialized value fully_connected/biases
fc = tf.contrib.layers.fully_connected(features,num_outputs=2)

sess=tf.InteractiveSession()
# It's required to initialize all the variables first
# or there'll be an error about precondition failures.
sess.run(tf.initialize_all_variables())
print(sess.run(fc))
