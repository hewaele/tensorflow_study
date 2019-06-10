import tensorflow as tf

#定义一个计算图
g1 = tf.Graph()
#再该计算图下面完成操作
with g1.as_default():
    #定义一个变量
    v = tf.get_variable('v', initializer=tf.zeros_initializer(), shape=[1, 2])

print('demo1')
#制定运行设备
with g1.device('/cpu:0'):
    with tf.Session(graph=g1) as sess:
        tf.global_variables_initializer().run()
        print(sess.run(v))

print('demo2')
#tf.variable_scope 和 get_varialbel 联合使用
with tf.variable_scope('g2'):
    v2 = tf.get_variable(name='v2', initializer=tf.zeros_initializer(), shape=[2, 3])
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(v2)
        print(sess.run(v2))

#tf.name_scope he Variable联合使用
print('demo3')
with tf.name_scope('g3'):
    #常数初始化
    v3 = tf.Variable(tf.constant([3, 4, 5], dtype=tf.float32), name='v3')
    #随机正太分布初始化
    v4 = tf.Variable(tf.random_normal(shape=[3], mean=1, stddev = 1), name='v4')
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(v3)
        print(v4)

        print(sess.run(v3))
        print(sess.run(v4))

print('demo4')
with tf.device('/cpu:0'):
    s = v3+v4
    s2 = tf.add(v3, v4, name='result')
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(s)
        print(sess.run(s))

        print(s2)
        print(sess.run(s2))


print('done')

