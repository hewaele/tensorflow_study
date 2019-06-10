#分类任务损失函数
import tensorflow as tf

pre = tf.constant([[-5, 3, 1000], [2, 1, 0]], dtype=tf.float32)
y_train = tf.constant([[0, 0, 1], [0, 1, 0]], dtype=tf.float32)

#调用官方
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_train, logits=pre)

#自己实现
def define_loss(pre, y_train):
    #转为概率
    s = tf.nn.softmax(pre)
    #避免概率越界
    s = tf.clip_by_value(s, 1e-10, 1)
    #计算交叉熵
    l = -tf.reduce_sum(y_train*tf.log(s), reduction_indices=1)
    #求平均
    cl = tf.reduce_mean(l)
    print(sess.run(s))
    print(sess.run(l))
    print(sess.run(cl))

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(loss))
    print('\n自己实现')
    define_loss(pre, y_train)

