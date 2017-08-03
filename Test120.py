#-*-coding=UTF-8-*-
import tensorflow as tf

'''
linear_regression 
python2.7
tf 0.11.0
'''
# initialize variables
W = tf.Variable(tf.zeros([2, 1], dtype=tf.float32), name="weights")
b = tf.Variable(0, dtype=tf.float32, name="bias")


# define the taining loop operations
def combine_inputs(X):
    return tf.matmul(X, W) + b


def inference(X):
    # return tf.sigmoid(combine_inputs(X))
    return tf.matmul(X, W) + b


def loss(X, Y):
    Y_ = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_), name="loss")


def inputs():
    # Data from http://people.sc.fsu.edu/~jburkardt/datasets/regression/x09.txt
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44],
                  [27, 24], [89, 31], [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34],
                  [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395, 434, 220, 374, 308,
                         220, 311, 181, 274, 303, 244]
    return tf.to_float(weight_age), tf.to_float(blood_fat_content)


def train(total_loss):
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):
    print sess.run(inference([[80., 25.]]))  # 203
    print sess.run(inference([[65., 25.]]))  # 256


saver = tf.train.Saver()

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss=total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    training_step = 10000
    for step in range(training_step):
        sess.run([train_op])

        if step % 10 == 0:
            print "Loss :", sess.run([total_loss])
        if step % 100 == 0:
            saver.save(sess, "./120/my_model", global_step=step)

    evaluate(sess, X, Y)

    saver.save(sess, "./120/my_model", global_step=training_step)

    coord.request_stop()
    coord.join(threads)
    sess.close()
