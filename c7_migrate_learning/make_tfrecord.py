import tensorflow as tf
import numpy as np
import os

labels_dic = {'daisy':0, 'dandelion':1, 'roses':2, 'sunflowers':3, 'tulips':4}


def creat_tfrecord(file_path, out_path):
    with tf.Session() as sess:
        #定义一个tfrecord wirter
        writer = tf.python_io.TFRecordWriter(out_path)
        for dir in os.listdir(file_path):
            if os.path.isdir(os.path.join(file_path, dir)):
                for img in os.listdir(os.path.join(file_path, dir)):
                    img_data = tf.gfile.FastGFile(os.path.join(file_path+dir, img), 'rb').read()
                    img_data = tf.image.decode_jpeg(img_data)
                    img_data = tf.image.resize_images(img_data, [299, 299])

                    #创建一个example
                    feature = tf.train.Features(feature={
                        'img':bytes_feature(bytes(img_data.eval())),
                        'label':int64_feature(labels_dic[dir])
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

    #解析一个样本
    features = tf.parse_single_example(example,
                                       features={
                                           'img':tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64)})

    #将解析出来的数据重新转换为原始数据
    image = tf.decode_raw(features['img'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)

    with tf.Session() as sess:
        print(sess.run(image))

def datasets_tfrecord(tfrecord_path):
    datasets = tf.data.TFRecordDataset([tfrecord_path])


def parse_function(example):
    pass


if __name__ == "__main__":
    file_path = './data/flower_photos/'
    out_path = './data/flowers.tfrecord'
    # creat_tfrecord(file_path, out_path)
    load_tfrecord(out_path)
