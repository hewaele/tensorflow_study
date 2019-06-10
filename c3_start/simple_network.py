import tensorflow as tf
from numpy.random import RandomState

x_train = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='x_train')
y_train = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='y_train')

with tf.name_scope('h1'):
    w1 = tf.Variable(tf.random_normal([2, 5], mean=0.2, stddev=1), dtype=tf.float32, name='w1')
    b1 = tf.Variable(tf.zeros([1, 5]), dtype=tf.float32, name='b1')
    h = tf.matmul(x_train, w1)
    h = tf.add(h, b1)

with tf.name_scope('h2'):
    w2 = tf.Variable(tf.random_normal([5, 3], mean=0.2, stddev=1), dtype=tf.float32, name='w2')
    b2 = tf.Variable(tf.zeros([1, 3]), dtype=tf.float32, name='b2')
    h2 = tf.matmul(h, w2)
    h2 = tf.add(h2, b2)

with tf.name_scope('h3'):
    w3 = tf.Variable(tf.random_normal([3, 1], mean=0.2, stddev=1), dtype=tf.float32, name='w1')
    b3 = tf.Variable(tf.zeros([1, 1]), dtype=tf.float32, name='b1')
    h3 = tf.matmul(h2, w3)

pre = tf.sigmoid(tf.add(h3, b3, name='out'))

#定义训练操作
loss = tf.reduce_mean(y_train*tf.log(tf.clip_by_value(pre, 1e-10, 1.0))+
                      (1-pre)*tf.log(tf.clip_by_value(1-pre, 1e-10, 1.0)))
tf.summary.scalar('loss', loss)
megred = tf.summary.merge_all()
write = tf.summary.FileWriter('./out_put/', tf.get_default_graph())

train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

#生成测试数据
x_data = RandomState(1).rand(512, 2)
y_data = [[int(x1+x2 < 1)] for x1, x2, in x_data]


with tf.Session() as sess:

    tf.global_variables_initializer().run()
    print(sess.run(b1))
    for i in range(100):
        _, s, total_loss = sess.run([train_op, megred, loss], feed_dict={x_train: x_data,
                                      y_train: y_data})
        # print(total_loss)
        write.add_summary(s, i)
    print(sess.run(b1))

print('done')
