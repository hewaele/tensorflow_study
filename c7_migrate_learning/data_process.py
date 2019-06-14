#使用npz格式在大数据集时，读取速度满效果不好
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from PIL import Image
import matplotlib.pyplot as plt
import os

data_image = []
data_label = []

photos_extentions = ['.jpg', '.JPG', '.jpeg', '.JPEG']
data_path = './data/flower_photos/'
count = 0

def creat_npy():
    for dir in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, dir)):
            for image in os.listdir(os.path.join(data_path, dir)):
                # img = tf.gfile.FastGFile(os.path.join(data_path + dir, image), 'rb').read()
                # image_data = tf.image.decode_jpeg(img)
                # #转换图片类型
                # if image_data.dtype != tf.float32:
                #     image_data = tf.image.convert_image_dtype(image_data, tf.float32)
                # #resize图片299*299  双线性插值
                # image_data = tf.image.resize_images(image_data, [299, 299])
                image_data = Image.open(os.path.join(data_path+dir, image))
                image_data = image_data.resize((299, 299))

                data_image.append(np.array(image_data))
                data_label.append(dir)



    #将数据进行随机切分
    train_x, test_x, train_y, test_y = train_test_split(data_image, data_label, test_size=0.2)

    print("{} {} {} {}".format(len(train_x), len(train_y), len(test_x), len(test_y)))
    #将数据保存
    train_valid_data = np.asarray([train_x, train_y, test_x, test_y])
    np.save('./data/train_valid_data.npy', train_valid_data)
    print('done')

def load_npy():
    data = np.load('./data/train_valid_data.npy', allow_pickle=True)
    train_x, test_x, train_y, test_y = data[0], data[1], data[2], data[3]
    print(len(train_y))
    print(len(train_x))

if __name__ == '__main__':
    # creat_npy()
    load_npy()