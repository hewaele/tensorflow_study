import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data

data_path = '/home/hewaele/PycharmProjects/tensorflow-study/c5_mnist/data/mnist.npz'

def conv(input, filter_shape):
    kernel = tf.Variable(tf.truncated_normal(shape=filter_shape, mean=0, stddev=0.1, dtype=tf.float32))
    b = tf.Variable(tf.zeros([1], dtype=tf.float32))
    c = tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding='SAME')
    result = tf.nn.relu(tf.nn.bias_add(c, b))

    return result

def build():
    pass


def train():
    pass


def inference(model, x):
    pass


if __name__ == '__main__':
    (train_x, train_y),(test_x, test_y) = load_data(data_path)

    input = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
    conv1 = conv(input, [3, 3, 1, 1])
    import numpy as np
    train_x = np.reshape(train_x, [-1, 28, 28, 1])
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        result = sess.run(conv1, feed_dict={input:train_x[:1]})
        print(result)


