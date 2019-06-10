#回归问题损失函数
#使用均方差
import tensorflow as tf

pre = tf.constant([[1], [3], [6]], dtype=tf.float32)
y_train = tf.constant([[1.5], [2.9], [10]], dtype=tf.float32)

#调用官方
loss = tf.reduce_mean(tf.square(pre-y_train))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(loss))