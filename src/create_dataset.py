from __future__ import division

import sys, os, time, math
import ipdb
import tensorflow as tf
import joblib
import numpy as np

from scipy.io import loadmat


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_dataset():
    def save_to_records(save_path, images, labels):
        # images: float32, labels: int32
        writer = tf.python_io.TFRecordWriter(save_path)
        for i in xrange(images.shape[0]):
            image_raw = images[i].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(32),
                'width': _int64_feature(32),
                'depth': _int64_feature(3),
                'label': _int64_feature(int(labels[i])),
                'image_raw': _bytes_feature(image_raw)
                }))
            writer.write(example.SerializeToString())

    data_root = '/home/bgshi/research/datasets/cifar-10-batches-py'
    train_images = np.zeros((50000,3072), dtype=np.float32)
    trian_labels = np.zeros((50000,), dtype=np.int32)
    for i in xrange(5):
        data_batch = joblib.load(os.path.join(data_root, 'data_batch_%d' % (i+1)))
        train_images[10000*i:10000*(i+1)] = data_batch['data'].astype(np.float32)
        trian_labels[10000*i:10000*(i+1)] = np.asarray(data_batch['labels'], dtype=np.int32)
    train_images = np.reshape(train_images, [50000,3,32,32])
    train_images = np.transpose(train_images, axes=[0,2,3,1])
    images_mean = np.mean(train_images, axis=0)
    data_std = np.std(train_images)
    train_images -= images_mean
    train_images /= data_std
    train_set = {'images': train_images, 'labels': trian_labels}
    save_to_records('../data/cifar10/train_simple_norm.tf', train_images, trian_labels)

    data_batch = joblib.load(os.path.join(data_root, 'test_batch'))
    test_images = data_batch['data'].astype(np.float32)
    test_images = np.reshape(test_images, [10000,3,32,32])
    test_images = np.transpose(test_images, axes=[0,2,3,1])
    test_images -= images_mean
    test_images /= data_std
    test_labels = np.asarray(data_batch['labels'], dtype=np.int32)
    test_set = {'images': test_images, 'labels': test_labels}
    save_to_records('../data/cifar10/test_simple_norm.tf', test_images, test_labels)
    

if __name__ == '__main__':
    create_dataset()
