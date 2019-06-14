import tensorflow as tf
import numpy as np
from PIL import Image
import os

labels_dic = {'daisy':0, 'dandelion':1, 'roses':2, 'sunflowers':3, 'tulips':4}


def creat_tfrecord(file_path, out_path):
    count = 0
    with tf.Session() as sess:
        #定义一个tfrecord wirter
        writer = tf.python_io.TFRecordWriter(out_path)
        for dir in os.listdir(file_path):
            if os.path.isdir(os.path.join(file_path, dir)):
                for img in os.listdir(os.path.join(file_path, dir)):
                    # img_data = tf.gfile.FastGFile(os.path.join(file_path+dir, img), 'rb').read()
                    # img_data = tf.image.decode_jpeg(img_data)
                    # img_data = tf.image.resize_images(img_data, [299, 299])
                    img_data = Image.open(os.path.join(file_path+dir, img))
                    img_data = img_data.resize((299, 299)).tobytes()

                    #创建一个example
                    feature = tf.train.Features(feature={
                        # 'img':bytes_feature(img_data),
                        'label': int64_feature(labels_dic[dir])
                    })
                    example = tf.train.Example(features=feature)
                    #将该张图片写入
                    writer.write(example.SerializeToString())

        writer.close()


#将图片像素转换为字符串类型
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


#将图片标签转换为整数型类型
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def load_tfrecord(tfrecord_path):
    #创建一个reader
    reader = tf.TFRecordReader()
    #创建一个文件队列
    file_queue = tf.train.string_input_producer([tfrecord_path])

    #从文件队列中对出一个样本
    _, example = reader.read(file_queue)
    #
    print('test')
    #解析一个样本
    features = tf.parse_single_example(example,
                                       features={
                                           'img':tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64)})
    print('test')
    #将解析出来的数据重新转换为原始数据
    image = tf.reshape(tf.decode_raw(features['img'], tf.uint8), [299, 299, 3])
    label = tf.cast(features['label'], tf.int32)
    with tf.Session() as sess:
        #启动队列线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        val = sess.run(image)
        print(len(val[0][0]))
    return image, label


def datasets_tfrecord(tfrecord_path):
    datasets = tf.data.TFRecordDataset([tfrecord_path])


def parse_function(example):
    pass

def tfrecord_test():
    tf_path = './data/tf_test.tfrecord'
    #随机生成一组数据用于测试
    a = [i for i in range(10)]

    #写入
    '''
    创建一个writer 
    循环数据
    穿件一个字典对象
    创建一个example实例，将数据字典传入
    将example写入
    '''
    writer = tf.python_io.TFRecordWriter(tf_path)
    for i in a:
        features = tf.train.Features(feature={'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))})
        example = tf.train.Example(features=features).SerializeToString()
        writer.write(example)

    writer.close()

    #读取
    reader = tf.TFRecordReader()
    sqeue = tf.train.string_input_producer([tf_path])
    _, example = reader.read(sqeue)
    features = tf.parse_single_example(example,features={'label':tf.FixedLenFeature([], tf.int64)})
    label = tf.cast(features['label'], tf.uint8)
    print(label)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10):
            print(sess.run(label))

if __name__ == "__main__":
    file_path = './data/flower_photos/'
    out_path = './data/flowers.tfrecord'
    # creat_tfrecord(file_path, out_path)
    image, label = load_tfrecord(out_path)
    tfrecord_test()
