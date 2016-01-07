from __future__ import division

import ipdb
import math
import tensorflow as tf
from tensorflow.python import control_flow_ops
import numpy as np

import model_utils as mu


FLAGS = tf.app.flags.FLAGS


def one_hot_embedding(label, n_classes):
    """
    One-hot embedding
    Args:
        label: int32 tensor [B]
        n_classes: int32, number of classes
    Return:
        embedding: tensor [B x n_classes]
    """
    embedding_params = np.eye(n_classes, dtype=np.float32)
    with tf.device('/cpu:0'):
        params = tf.constant(embedding_params)
        embedding = tf.gather(params, label)
    return embedding


def conv2d(x, n_in, n_out, k, s, p='SAME', bias=False, scope='conv'):
    with tf.variable_scope(scope):
        kernel = tf.Variable(
            tf.truncated_normal([k, k, n_in, n_out],
                stddev=math.sqrt(2/(k*k*n_out))),
            name='weight')
        tf.add_to_collection('weights', kernel)
        conv = tf.nn.conv2d(x, kernel, [1,s,s,1], padding=p)
        if bias:
            bias = tf.Variable(tf.zeros([n_out]), name='bias')
            tf.add_to_collection('biases', bias)
            conv = tf.nn.bias_add(conv, bias)
    return conv


def batch_norm(x, n_out, phase_train, scope='bn', affine=True):
    """
    Batch normalization on convolutional maps.
    Args:
        x: Tensor, 4D BHWD input maps
        n_out: integer, depth of input maps
        phase_train: boolean tf.Variable, true indicates training phase
        scope: string, variable scope
        affine: whether to affine-transform outputs
    Return:
        normed: batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
            name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
            name='gamma', trainable=affine)
        tf.add_to_collection('biases', beta)
        tf.add_to_collection('weights', gamma)

        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = control_flow_ops.cond(phase_train,
            mean_var_with_update,
            lambda: (ema_mean, ema_var))

        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
            beta, gamma, 1e-3, affine)
    return normed


def residual_block(x, n_in, n_out, subsample, phase_train, scope='res_block'):
    with tf.variable_scope(scope):
        if subsample:
            y = conv2d(x, n_in, n_out, 3, 2, 'SAME', False, scope='conv_1')
            shortcut = conv2d(x, n_in, n_out, 3, 2, 'SAME', False, scope='shortcut')
        else:
            y = conv2d(x, n_in, n_out, 3, 1, 'SAME', False, scope='conv_1')
            shortcut = tf.identity(x, name='shortcut')
        y = batch_norm(y, n_out, phase_train, scope='bn_1')
        y = tf.nn.relu(y, name='relu_1')
        y = conv2d(y, n_out, n_out, 3, 1, 'SAME', True, scope='conv_2')
        y = y + shortcut
        y = batch_norm(y, n_out, phase_train, scope='bn_2')
        y = tf.nn.relu(y, name='relu_2')
    return y


def residual_group(x, n_in, n_out, n, first_subsample, phase_train, scope='res_group'):
    with tf.variable_scope(scope):
        y = residual_block(x, n_in, n_out, first_subsample, phase_train, scope='block_1')
        for i in xrange(n-1):
            y = residual_block(y, n_out, n_out, False, phase_train, scope='block_%d' % (i+2))
    return y


def residual_net(x, n, n_classes, phase_train, scope='res_net'):
    with tf.variable_scope(scope):
        mu.activation_summary(x)
        y = conv2d(x, 3, 16, 3, 1, 'SAME', False, scope='conv_init')
        y = batch_norm(y, 16, phase_train, scope='bn_init')
        y = tf.nn.relu(y, name='relu_init')
        mu.activation_summary(y)
        y = residual_group(y, 16, 16, n, False, phase_train, scope='group_1')
        mu.activation_summary(y)
        y = residual_group(y, 16, 32, n, True, phase_train, scope='group_2')
        mu.activation_summary(y)
        y = residual_group(y, 32, 64, n, True, phase_train, scope='group_3')
        mu.activation_summary(y)
        y = conv2d(y, 64, n_classes, 1, 1, 'SAME', True, scope='conv_last')
        mu.activation_summary(y)
        y = tf.nn.avg_pool(y, [1,8,8,1], [1,1,1,1], 'VALID', name='avg_pool')
        y = tf.squeeze(y, squeeze_dims=[1,2])
    return y


def loss(logits, labels, scope='loss'):
    with tf.variable_scope(scope):
        # entropy loss
        targets = one_hot_embedding(labels, 10)
        entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, targets),
            name='entropy_loss')
        tf.add_to_collection('losses', entropy_loss)

        # weight l2 decay loss
        weight_l2_losses = [tf.nn.l2_loss(o) for o in tf.get_collection('weights')]
        weight_decay_loss = tf.mul(FLAGS.weight_decay, tf.add_n(weight_l2_losses),
            name='weight_decay_loss')
        tf.add_to_collection('losses', weight_decay_loss)

        # total loss
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    for var in tf.get_collection('losses'):
        tf.scalar_summary('losses/' + var.op.name, var)
    return total_loss


def accuracy(logits, gt_label, scope='accuracy'):
    with tf.variable_scope(scope):
        pred_label = tf.argmax(logits, 1)
        acc = 1.0 - tf.nn.zero_fraction(
            tf.cast(tf.equal(pred_label, gt_label), tf.int32))
    return acc


def train_op(loss, global_step, learning_rate):
    tf.scalar_summary('learning_rate', learning_rate)
    learning_rate_weights = learning_rate
    learning_rate_biases = 2.0 * learning_rate # double learning rate for biases

    weights, biases = tf.get_collection('weights'), tf.get_collection('biases')
    assert(len(weights) + len(biases) == len(tf.trainable_variables()))
    params = weights + biases
    gradients = tf.gradients(loss, params, name='gradients')
    gradient_weights = gradients[:len(weights)]
    gradient_biases = gradients[len(weights):]

    optim_weights = tf.train.MomentumOptimizer(learning_rate_weights, 0.9)
    optim_biases = tf.train.MomentumOptimizer(learning_rate_biases, 0.9)
    update_weights = optim_weights.apply_gradients(zip(gradient_weights, weights))
    update_biases = optim_biases.apply_gradients(zip(gradient_biases, biases))

    with tf.control_dependencies([update_weights, update_biases]):
        train_op = tf.no_op(name='train_op')
    return train_op


def cifar10_input_stream(records_path):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([records_path], None)
    _, record_value = reader.read(filename_queue)
    features = tf.parse_single_example(
        record_value,
        dense_keys=['image_raw', 'label'],
        dense_types=[tf.string, tf.int64])
    image = tf.decode_raw(features['image_raw'], tf.float32)
    image = tf.reshape(image, [32,32,3])
    label = tf.cast(features['label'], tf.int64)
    return image, label


def random_distort_image(image):
    distorted_image = image
    distorted_image = tf.image.pad_to_bounding_box(image, 4, 4, 40, 40) # pad 4 pixels to each side
    distorted_image = tf.image.random_crop(distorted_image, [32, 32])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    # distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    # distorted_image = tf.image.per_image_whitening(distorted_image)
    return distorted_image


def make_train_batch(train_records_path, batch_size):
    with tf.variable_scope('train_batch'):
        with tf.device('/cpu:0'):
            train_image, train_label = cifar10_input_stream(train_records_path)
            train_image = random_distort_image(train_image)
            train_image_batch, train_label_batch = tf.train.shuffle_batch(
                [train_image, train_label], batch_size=batch_size, num_threads=4,
                capacity=50000,
                min_after_dequeue=1000)
    return train_image_batch, train_label_batch


def make_validation_batch(test_records_path, batch_size):
    with tf.variable_scope('evaluate_batch'):
        with tf.device('/cpu:0'):
            test_image, test_label = cifar10_input_stream(test_records_path)
            test_image_batch, test_label_batch = tf.train.batch(
                [test_image, test_label], batch_size=batch_size, num_threads=1,
                capacity=10000)
    return test_image_batch, test_label_batch
