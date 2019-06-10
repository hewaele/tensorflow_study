import tensorflow as tf

class Model():
    """
    参数
    net_shape 网络结构层数
    lr 学习率
    decay    学习率衰减
    """
    def __init__(self, net_shape=None, lr=0.005, decay=0.9, epoch=100, batch=50, save_path=''):
        self.net_shape = net_shape
        self.lr = lr
        self.decay = decay
        self.epoch = epoch
        self.batch = batch
        self.save_path = save_path

        self.build()

    """
    构建网络
    """
    def build(self):
        self.input = tf.placeholder(shape=[None, self.net_shape[0]], dtype=tf.float32, name='input')
        self.output = tf.placeholder(shape=[None, self.net_shape[-1]], dtype=tf.float32, name='output')

        front = self.input
        for index, hi in enumerate(self.net_shape[:-1]):
            w = tf.Variable(tf.truncated_normal(shape=[hi, self.net_shape[index+1]], dtype=tf.float32, stddev=0.1))
            b = tf.Variable(tf.zeros(shape=[self.net_shape[index+1]], dtype=tf.float32))

            h = tf.nn.relu(tf.add(tf.matmul(front, w), b))
            front = h

        self.pre = tf.nn.softmax(h, name='predict')


    """
    构建训练过程
    """
    def train(self, train_x, train_y):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.output, logits=self.pre))
        acc = tf.equal(tf.argmax(self.pre, 1), tf.argmax(self.output, 1))
        acc = tf.reduce_mean(tf.cast(acc, dtype=tf.float32))
        op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        #将训练数据转换为datasets格式
        dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(self.batch)
        dataset = dataset.make_one_shot_iterator()
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(self.epoch):
                x, y = dataset.get_next()
                x, y = sess.run([x, y])
                y = [[int(j==i) for j in range(10)] for i in y]
                _, l, point = sess.run([op, loss, acc], feed_dict={self.input: x, self.output: y})
                print('epoch{}: loss:{} acc:{}'.format(i, l, point))
        print('trian done')
        self.save_model()

    def inference(self, x):
        model = self.load_model()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()


    def save_model(self):
        pass


    def load_model(self):

        return


def load_data(path='/home/hewaele/PycharmProjects/tensorflow-study/c5_mnist/data/mnist.npz'):
    from tensorflow.keras.datasets import mnist
    import numpy as np

    (train_x, train_y), (test_x, test_y) = mnist.load_data(path)
    train_x = np.reshape(train_x, (-1, 28*28))/255
    test_x = np.reshape(test_x, (-1, 28*28))/255

    return train_x, train_y, test_x, test_y

if __name__ == "__main__":
    mnist = Model(net_shape=[28*28, 500, 200, 10])
    train_x, train_y, test_x, test_y = load_data()
    mnist.train(train_x, train_y)



