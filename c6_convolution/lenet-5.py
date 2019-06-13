import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.mnist import load_data

#定义一个模型类
class LENET():
    """

    """
    def __init__(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(0.1, self.global_step, 1000, 0.95, staircase=True)
        self.epoch = 50000
        self.batch = 128

        self.build()

    def build(self):
        self.x = tf.placeholder(shape=(None, 28, 28, 1), dtype=tf.float32, name='input')
        self.y = tf.placeholder(shape=(None, 10), dtype=tf.float32, name='y')

        # with tf.name_scope('conv1'):
        #     c = self.conv(self.x, [5, 5, 1, 6], step=[1, 1, 1, 1], pad='VALID')
        #     c = tf.nn.max_pool(c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #
        # with tf.name_scope('conv2'):
        #     c = self.conv(c, [5, 5, 6, 16], step=[1, 1, 1, 1], pad='VALID')
        #     c = tf.nn.max_pool(c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.name_scope('fc1'):
            c = tf.reshape(self.x, shape=[-1, 28*28])
            w = tf.Variable(tf.random_normal(shape=[784, 512], dtype=tf.float32, stddev=0.1))
            b = tf.Variable(tf.random_normal(shape=[512], dtype=tf.float32, stddev=0.1))
            f = tf.add(tf.matmul(c, w), b)
            f = tf.nn.relu(f)

        with tf.name_scope('fc2'):
            w = tf.Variable(tf.random_normal(shape=[512, 156], dtype=tf.float32, stddev=0.1))
            b = tf.Variable(tf.random_normal(shape=[156], dtype=tf.float32, stddev=0.1))
            f = tf.add(tf.matmul(f, w), b)
            f = tf.nn.relu(f)
            # f = tf.nn.dropout(f, 0.5)

        with tf.name_scope('fc3'):
            w = tf.Variable(tf.random_normal(shape=[156, 10], dtype=tf.float32, stddev=0.1))
            b = tf.Variable(tf.random_normal(shape=[10], dtype=tf.float32, stddev=0.1))
            f = tf.matmul(f, w) + b

        self.pre = tf.nn.softmax(f, name='predict')


    def conv(self, x, kernel_shape, step, pad):
        kernel_weight = tf.Variable(tf.random_normal(kernel_shape, dtype=tf.float32, stddev=0.1))
        bias = tf.Variable(tf.random_normal([kernel_shape[-1]], dtype=tf.float32))
        c = tf.nn.conv2d(x, kernel_weight, strides=step, padding=pad)
        c = tf.nn.relu(tf.nn.bias_add(c, bias))
        return c


    def train(self, train_iter, test_iter, vx, vy):
        cross_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.pre))
        op = tf.train.GradientDescentOptimizer(0.05).minimize(cross_loss)
        accuracy = tf.equal(tf.argmax(self.y, axis=1), tf.argmax(self.pre, axis=1))
        accuracy = tf.reduce_mean(tf.cast(accuracy, dtype=tf.float32))

        tf.summary.scalar('lr', self.lr)
        tf.summary.scalar('loss', cross_loss)
        tf.summary.scalar('ac', accuracy)
        merge = tf.summary.merge_all()
        write = tf.summary.FileWriter('./log/', tf.get_default_graph())

        tx, ty = train_iter.get_next()
        # vx, vy = test_iter.get_next()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(self.epoch):
                # print(tx)
                _, loss, m = sess.run([op, cross_loss, merge], feed_dict={self.x: sess.run(tx), self.y: sess.run(ty)})
                write.add_summary(m, i)

                if i % 1000 == 0:
                    p, loss, acc = sess.run([self.pre, cross_loss, accuracy],
                                            feed_dict={self.x: sess.run(vx), self.y: sess.run(vy)})
                    print('epoch:{} loss:{} ac:{}'.format(i, loss, acc))
                    print(p[-1])
                    print(sess.run(vy)[-1])

    def inference(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

def data_pre(data_path):
    (train_x, train_y), (test_x, test_y) = load_data(data_path)
    print(len(train_x))
    print(len(test_x))
    #将数据转换为小数
    train_x = train_x/255.0
    test_x = test_x/255.0

    #将label独热
    train_x = tf.cast(tf.reshape(train_x, [-1, 28, 28, 1]), dtype=tf.float32)
    test_x = tf.cast(tf.reshape(test_x, [-1, 28, 28, 1]), dtype=tf.float32)

    train_y = tf.one_hot(train_y, dtype=tf.float32, depth=10)
    test_y = tf.one_hot(test_y, dtype=tf.float32, depth=10)

    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))

    train_data = train_data.repeat(200).shuffle(100).batch(128)
    train_iter = train_data.make_one_shot_iterator()

    test_data = test_data.repeat(10).shuffle(1000).batch(500)
    test_iter = test_data.make_one_shot_iterator()

    return train_iter, test_iter, test_x, test_y


if __name__ == "__main__":
    data_path = '/home/hewaele/PycharmProjects/tensorflow-study/c5_mnist/data/mnist.npz'
    train_iter, test_iter, vx, vy = data_pre(data_path)

    #构建网络
    net = LENET()
    #执行训练
    net.train(train_iter, test_iter, vx, vy)
