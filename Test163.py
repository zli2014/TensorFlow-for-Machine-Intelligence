#-*-coding=UTF-8-*-

import tensorflow as tf
import matplotlib as mil
mil.use("svg")
#mil.use("nbagg")
from matplotlib import pyplot
fig = pyplot.gcf()
fig.set_size_inches(4,4)

image_filename="./images/chapter-05-object-recognition-and-classification/convolution/n02113023_219.jpg"

''' 
    tf.train.string_input_producer(string_tensor, 
    num_epochs=None, shuffle=True, seed=None, 
    capacity=32, shared_name=None, 
    name=None, cancel_op=None)
    可以将图像或者是文件转化成tensor 对象  
    string_tensor 是必要的参数
    
    tf.train.match_filenames_once(pattern, name=None) :
            将模式匹配成功的文件列表保存下来
            参数 pattern 为匹配模式
            return : 返回的是匹配成功的文件列表
    
'''
filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(image_filename))

print('filename_queue')
print(filename_queue)
'''
Class
A Reader that outputs the entire contents of a file as a value.
To use, enqueue filenames in a Queue. 
The output of Read will be a filename (key) and the contents of that file (value).
'''
image_reader = tf.WholeFileReader()
'''
tf.WholeFileReader.read(queue, name=None)
返回 A tuple of Tensors (key, value). 
'''
_, image_file = image_reader.read(filename_queue)
image = tf.image.decode_jpeg(image_file)

sess = tf.InteractiveSession()

sess.run(tf.initialize_all_variables())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

image_batch = tf.image.convert_image_dtype(tf.expand_dims(image, 0), tf.float32, saturate=False)

kernel = tf.constant([
        [
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]]
        ],
        [
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
            [[ 8., 0., 0.], [ 0., 8., 0.], [ 0., 0., 8.]],
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]]
        ],
        [
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]],
            [[ -1., 0., 0.], [ 0., -1., 0.], [ 0., 0., -1.]]
        ]
    ])


conv2d = tf.nn.conv2d(image_batch, kernel, [1, 1, 1, 1], padding="SAME")
activation_map = sess.run(tf.minimum(tf.nn.relu(conv2d), 255))

# setup-only-ignore
fig = pyplot.gcf()
pyplot.imshow(activation_map[0], interpolation='nearest')
#pyplot.show()
fig.set_size_inches(4, 4)
fig.savefig("./images/chapter-05-object-recognition-and-classification/convolution/example-edge-detection.png")
