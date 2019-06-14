import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import os

def parse_data(example):
    features = {'img':tf.FixedLenFeature([], tf.string),
                'label':tf.FixedLenFeature([], tf.int64)}
    example = tf.parse_single_example(example, features=features)
    example['img'] = tf.reshape(tf.decode_raw(example['img'], tf.uint8), [299, 299, 3])
    example['label'] = tf.cast(example['label'], tf.int32)

    return example

def tfrecord_datasets(tfrecord_path):
    dataset = tf.data.TFRecordDataset([tfrecord_path])
    data = dataset.map(parse_data).shuffle(5000).repeat(50).batch(64)
    next_data = data.make_one_shot_iterator().get_next()

    # with tf.Session() as sess:
    #     tf.global_variables_initializer().run()
    #     print(next_data['img'])
    #     print(sess.run(next_data['label']))
    #     print(sess.run(next_data['label']))
    return next_data


def main():
    tfrecord_path = './data/flowers.tfrecord'
    next_data = tfrecord_datasets(tfrecord_path)
    model = keras.applications.InceptionV3(include_top=False,
                                           input_tensor=tf.cast(next_data['img'], tf.float32),
                                           input_shape=[299, 299, 3])

    last = model.output
    x = keras.layers.GlobalAveragePooling2D()(last)
    x = keras.layers.Dense(1000, activation='relu')(x)
    out = keras.layers.Dense(10, activation='softmax')(x)
    my_inception = keras.Model(model.input, out)
    my_inception.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    print(my_inception.summary())
    my_inception.fit(x=tf.cast(next_data['img'], tf.float32),
                     y=tf.one_hot(next_data['label'], depth=10),
                     steps_per_epoch=1000)

if __name__ == '__main__':
    main()
