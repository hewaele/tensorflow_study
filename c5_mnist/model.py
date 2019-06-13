import tensorflow as tf

class Model():
    """
    参数
    net_shape 网络结构层数
    lr 学习率
    decay    学习率衰减
    """
    def __init__(self, x, y, net_shape=None, lr=0.0005, decay=0.9, epoch=30000, batch=100, save_path='./output/model.ckpt'):
        self.net_shape = net_shape
        self.lr = lr
        self.decay = decay
        self.epoch = epoch
        self.batch = batch
        self.save_path = save_path
        self.input = x
        self.output = y

        self.build()

    """
    构建网络
    """
    def build(self):
        front = self.input
        for index, hi in enumerate(self.net_shape[:-1]):
            w = tf.Variable(tf.truncated_normal(shape=[hi, self.net_shape[index+1]], dtype=tf.float32, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[self.net_shape[index+1]], dtype=tf.float32))

            h = tf.nn.relu(tf.add(tf.matmul(front, w), b))
            front = h

        self.pre = tf.nn.softmax(h, name='predict')


    """
    构建训练过程
    """
    def train(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.output, logits=self.pre))
        acc = tf.equal(tf.argmax(self.pre, 1), tf.argmax(self.output, 1))
        acc = tf.reduce_mean(tf.cast(acc, dtype=tf.float32))
        op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(self.epoch):
                sess.run([op])

                if i%500 == 0:
                    l, point = sess.run([loss, acc])
                    print('epoch{}: loss:{} acc:{}'.format(i, l, point))

            print('trian done')
            self.save_model(sess, self.save_path)

    def inference(self, x):
        model = self.load_model()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            sess.run(model, feed_dict={'input:0': x})

    def save_model(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, path)


    def load_model(self):
        saver = tf.train.import_meta_graph(self.save_path+'.meta')
        with tf.Session() as sess:
            saver.restore(sess, self.save_path)
            predict = tf.get_default_graph().get_tensor_by_name('input:0')
        return predict

    def load_model2(self):
        model = self.build()

def load_data(path='/home/hewaele/PycharmProjects/tensorflow-study/c5_mnist/data/mnist.npz'):
    from tensorflow.keras.datasets import mnist
    import numpy as np

    (train_x, train_y), (test_x, test_y) = mnist.load_data(path)
    train_x = np.reshape(train_x, (-1, 28*28))/255
    test_x = np.reshape(test_x, (-1, 28*28))/255

    return train_x, train_y, test_x, test_y

if __name__ == "__main__":
    import numpy as np

    train_x, train_y, test_x, test_y = load_data()

    train_x = train_x.astype(np.float32)
    train_y = [[float(v == j) for j in range(10)] for v in train_y]
    test_y = [[float(v == j) for j in range(10)] for v in test_y]

    # 将训练数据转换为datasets格式
    dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    dataset = dataset.repeat(100)
    dataset = dataset.shuffle(100).batch(50)
    it = dataset.make_one_shot_iterator()
    x, y = it.get_next()
    dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    dataset = dataset.shuffle(1000).batch(500)
    it = dataset.make_one_shot_iterator()
    tx, ty = it.get_next()
    x = tf.identity(x, name='inpyt')
    print(x)
    mnist = Model(x, y, net_shape=[28 * 28, 500, 200, 100, 10])
    mnist.inference(tx)



