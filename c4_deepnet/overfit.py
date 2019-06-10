#使用l2正则解决过拟合
import tensorflow as tf

pre = tf.constant(tf.random_normal(shape=[1, 5], dtype=tf.float32), name='predict')
train = tf.constant(tf.random_normal(shape=[1, 5], dtype=tf.float32), name='real')
